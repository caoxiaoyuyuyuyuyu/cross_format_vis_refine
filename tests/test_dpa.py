"""Unit tests for DPA module."""

import pytest
import torch
import torch.nn.functional as F

from src.model.dpa import DifferentialPerceptionAdapter, CrossAttentionLayer


# ─── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def dpa():
    return DifferentialPerceptionAdapter()


@pytest.fixture
def feature_pair():
    """Simulated hook features for batch_size=2, seq_len=256, dim=1280."""
    B, S, D = 2, 256, 1280
    layers = (7, 23, 31)
    target = {l: torch.randn(B, S, D) for l in layers}
    rendered = {l: torch.randn(B, S, D) for l in layers}
    return target, rendered


# ─── DPA Forward Shape ─────────────────────────────────────────────

def test_dpa_forward_shape(dpa, feature_pair):
    """Verify DPA output shape: (B, S, llm_hidden_dim)."""
    target, rendered = feature_pair
    diff_tokens = dpa(target, rendered)
    assert diff_tokens.shape == (2, 256, 3584)


def test_dpa_forward_single_sample(dpa):
    """Test with batch_size=1."""
    B, S, D = 1, 100, 1280
    target = {l: torch.randn(B, S, D) for l in (7, 23, 31)}
    rendered = {l: torch.randn(B, S, D) for l in (7, 23, 31)}
    diff_tokens = dpa(target, rendered)
    assert diff_tokens.shape == (1, 100, 3584)


# ─── Parameter Count ───────────────────────────────────────────────

def test_dpa_param_count(dpa):
    """Verify ~12M trainable parameters."""
    count = dpa.get_param_count()
    assert 10_000_000 < count < 15_000_000, f"Param count {count} outside 10M-15M range"
    print(f"\nDPA param count: {count:,}")


# ─── Gradient Flow ─────────────────────────────────────────────────

def test_dpa_gradient_flow(dpa, feature_pair):
    """Ensure gradients flow back through all DPA parameters."""
    target, rendered = feature_pair
    # Make features require grad to test full backward
    for d in (target, rendered):
        for v in d.values():
            v.requires_grad_(True)

    diff_tokens = dpa(target, rendered)
    loss = diff_tokens.sum()
    loss.backward()

    # All DPA parameters should have gradients
    for name, param in dpa.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


# ─── Diff Feature Extraction ──────────────────────────────────────

def test_extract_diff_features(dpa, feature_pair):
    """Verify diff features are computed correctly."""
    target, rendered = feature_pair
    diffs = dpa.extract_diff_features(target, rendered)

    assert len(diffs) == 3
    for d in diffs:
        assert d.shape == (2, 256, 1280)


def test_diff_identical_images(dpa):
    """When images are identical, diff features should be zero."""
    B, S, D = 1, 64, 1280
    feats = {l: torch.randn(B, S, D) for l in (7, 23, 31)}
    diffs = dpa.extract_diff_features(feats, feats)

    for d in diffs:
        assert torch.allclose(d, torch.zeros_like(d), atol=1e-6)


# ─── Cross-Attention Layer ─────────────────────────────────────────

def test_cross_attention_shape():
    """Verify cross-attention output matches query shape."""
    layer = CrossAttentionLayer(hidden_dim=1280, inner_dim=768, num_heads=16)
    q = torch.randn(2, 256, 1280)
    kv = torch.randn(2, 768, 1280)  # 3x query length (concat of 3 scales)
    out = layer(q, kv)
    assert out.shape == q.shape


# ─── Fusion ────────────────────────────────────────────────────────

def test_fuse_features_shape(dpa):
    """Verify fusion output shape."""
    B, S, D = 2, 256, 1280
    target_feat = torch.randn(B, S, D)
    diff_features = [torch.randn(B, S, D) for _ in range(3)]
    fused = dpa.fuse_features(target_feat, diff_features)
    assert fused.shape == (B, S, D)


# ─── Determinism ───────────────────────────────────────────────────

def test_deterministic_output(dpa, feature_pair):
    """Same input should produce same output."""
    target, rendered = feature_pair
    dpa.eval()
    with torch.no_grad():
        out1 = dpa(target, rendered)
        out2 = dpa(target, rendered)
    assert torch.allclose(out1, out2)


# ─── Integration: 2D→3D reshape (simulating real ViT hook output) ───

def test_dpa_with_2d_hook_output_uniform(dpa):
    """Simulate real Qwen2.5-VL ViT hook: 2D (total_patches, D) with uniform image sizes.

    This tests the _encode_image reshape path in diffcode.py.
    DPA itself expects 3D, so this test verifies the reshape logic externally.
    """
    B, S, D = 2, 256, 1280
    layers = (7, 23, 31)

    # Simulate 2D hook output: (B*S, D)
    raw_target = {l: torch.randn(B * S, D) for l in layers}
    raw_rendered = {l: torch.randn(B * S, D) for l in layers}

    # Reshape to 3D as _encode_image would do
    target_3d = {l: raw_target[l].view(B, S, D) for l in layers}
    rendered_3d = {l: raw_rendered[l].view(B, S, D) for l in layers}

    diff_tokens = dpa(target_3d, rendered_3d)
    assert diff_tokens.shape == (B, S, 3584)


def test_dpa_with_2d_hook_output_variable(dpa):
    """Simulate variable-length images: different patch counts per image.

    Tests the pad-and-stack path in _encode_image.
    """
    D = 1280
    layers = (7, 23, 31)
    seq_lens = [200, 300]  # Two images with different patch counts
    total = sum(seq_lens)
    max_len = max(seq_lens)

    raw_target = {l: torch.randn(total, D) for l in layers}
    raw_rendered = {l: torch.randn(total, D) for l in layers}

    # Reshape with split-and-pad as _encode_image would do
    target_3d = {}
    rendered_3d = {}
    for l in layers:
        for raw, out_dict in [(raw_target, target_3d), (raw_rendered, rendered_3d)]:
            splits = raw[l].split(seq_lens, dim=0)
            padded = []
            for s in splits:
                pad_len = max_len - s.shape[0]
                if pad_len > 0:
                    s = F.pad(s, (0, 0, 0, pad_len))
                padded.append(s)
            out_dict[l] = torch.stack(padded, dim=0)

    diff_tokens = dpa(target_3d, rendered_3d)
    assert diff_tokens.shape == (2, max_len, 3584)
