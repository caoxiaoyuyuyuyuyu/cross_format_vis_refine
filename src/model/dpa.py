"""Differential Perception Adapter (DPA).

Core module of DiffCode. Inserted after Qwen2.5-VL ViT to extract
multi-scale diff features from full-attention layers and project them
into LLM input space as explicit "where is the difference" signal.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CrossAttentionLayer(nn.Module):
    """Bottleneck cross-attention: projects to inner_dim for attention, then back."""

    def __init__(self, hidden_dim: int, inner_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = inner_dim // num_heads
        assert inner_dim % num_heads == 0

        self.q_proj = nn.Linear(hidden_dim, inner_dim)
        self.k_proj = nn.Linear(hidden_dim, inner_dim)
        self.v_proj = nn.Linear(hidden_dim, inner_dim)
        self.o_proj = nn.Linear(inner_dim, hidden_dim)
        self.layer_norm_q = nn.LayerNorm(hidden_dim)
        self.layer_norm_kv = nn.LayerNorm(hidden_dim)

    def forward(self, query: Tensor, key_value: Tensor) -> Tensor:
        q = self.layer_norm_q(query)
        kv = self.layer_norm_kv(key_value)

        B, Sq, _ = q.shape
        _, Skv, _ = kv.shape

        q = self.q_proj(q).view(B, Sq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(B, Skv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, Skv, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(B, Sq, -1)
        out = self.o_proj(attn)

        return query + out


class DifferentialPerceptionAdapter(nn.Module):
    """DPA: extracts multi-scale visual diff features and projects to LLM space.

    Architecture:
        1. Element-wise subtraction at each hook layer -> diff features
        2. Bottleneck cross-attention fusion (target queries, diff key/values)
        3. Linear projection to LLM hidden dim

    Parameter budget (~12M):
        - 2 cross-attn layers (1280->768 bottleneck): ~7.9M
        - Projection (LN + Linear 1280->3584): ~4.6M
        - Scale weights: 3
    """

    def __init__(
        self,
        vit_hidden_dim: int = 1280,
        llm_hidden_dim: int = 3584,
        num_heads: int = 16,
        num_cross_attn_layers: int = 2,
        inner_dim: int = 768,
        hook_layers: Tuple[int, ...] = (7, 23, 31),
    ):
        super().__init__()
        self.vit_hidden_dim = vit_hidden_dim
        self.llm_hidden_dim = llm_hidden_dim
        self.hook_layers = hook_layers
        self.num_scales = len(hook_layers)

        # Per-scale learnable weights for diff features
        self.scale_weights = nn.Parameter(torch.ones(self.num_scales))

        # Bottleneck cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(vit_hidden_dim, inner_dim, num_heads)
            for _ in range(num_cross_attn_layers)
        ])

        # Project fused features to LLM input space
        self.projection = nn.Sequential(
            nn.LayerNorm(vit_hidden_dim),
            nn.Linear(vit_hidden_dim, llm_hidden_dim),
        )

    def extract_diff_features(
        self, target_features: Dict[int, Tensor], rendered_features: Dict[int, Tensor]
    ) -> List[Tensor]:
        """Extract per-layer diff features via element-wise subtraction."""
        diff_features = []
        weights = F.softmax(self.scale_weights, dim=0)
        for i, layer_idx in enumerate(self.hook_layers):
            diff = target_features[layer_idx] - rendered_features[layer_idx]
            diff_features.append(diff * weights[i])
        return diff_features

    def fuse_features(
        self, target_feat: Tensor, diff_features: List[Tensor]
    ) -> Tensor:
        """Cross-attention fusion of target features with multi-scale diffs."""
        kv = torch.cat(diff_features, dim=1)
        x = target_feat
        for layer in self.cross_attn_layers:
            x = layer(x, kv)
        return x

    def forward(
        self, target_features: Dict[int, Tensor], rendered_features: Dict[int, Tensor]
    ) -> Tensor:
        """Full forward: diff extraction -> fusion -> projection -> diff_tokens."""
        diff_features = self.extract_diff_features(target_features, rendered_features)
        target_feat = target_features[self.hook_layers[-1]]
        fused = self.fuse_features(target_feat, diff_features)
        diff_tokens = self.projection(fused)
        return diff_tokens

    def get_param_count(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
