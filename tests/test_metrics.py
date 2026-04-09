"""Tests for evaluation metrics."""

import sys
import os
import numpy as np
import pytest
from PIL import Image
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.evaluation.metrics import MetricsComputer


@pytest.fixture
def mc():
    return MetricsComputer(device="cpu")


def _make_image(color, size=(64, 64)):
    """Create a solid color image."""
    arr = np.full((*size, 3), color, dtype=np.uint8)
    return Image.fromarray(arr)


class TestSSIM:
    def test_identical_images(self, mc):
        img = _make_image((128, 128, 128))
        score = mc.compute_ssim(img, img)
        assert score > 0.99, f"Identical images should have SSIM ~1.0, got {score}"

    def test_different_images(self, mc):
        img_a = _make_image((0, 0, 0))
        img_b = _make_image((255, 255, 255))
        score = mc.compute_ssim(img_a, img_b)
        assert score < 0.5, f"Very different images should have low SSIM, got {score}"

    def test_numpy_input(self, mc):
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        score = mc.compute_ssim(arr, arr)
        assert score > 0.99

    def test_different_sizes(self, mc):
        img_a = _make_image((100, 100, 100), size=(64, 64))
        img_b = _make_image((100, 100, 100), size=(128, 128))
        score = mc.compute_ssim(img_a, img_b)
        assert 0.0 <= score <= 1.0


class TestCLIP:
    """CLIP tests - these will skip if model download is not available."""

    def test_identical_images(self, mc):
        img = _make_image((128, 128, 128))
        score = mc.compute_clip_score(img, img)
        if score < 0:
            pytest.skip("CLIP model not available")
        assert score > 0.8, f"Identical images should have high CLIP score, got {score}"

    def test_different_images(self, mc):
        np.random.seed(42)
        arr_a = np.random.randint(0, 128, (224, 224, 3), dtype=np.uint8)
        arr_b = np.random.randint(128, 255, (224, 224, 3), dtype=np.uint8)
        img_a = Image.fromarray(arr_a)
        img_b = Image.fromarray(arr_b)
        score = mc.compute_clip_score(img_a, img_b)
        if score < 0:
            pytest.skip("CLIP model not available")
        assert score < 0.99

    def test_failure_returns_negative(self, mc):
        """When CLIP model fails to load, should return -1.0."""
        mc._clip_failed = True  # Simulate failed load
        img = _make_image((128, 128, 128))
        score = mc.compute_clip_score(img, img)
        assert score == -1.0


class TestCodeBLEU:
    def test_identical_code(self, mc):
        code = "<div><p>Hello World</p></div>"
        score = mc.compute_codebleu(code, code)
        assert score > 0.9, f"Identical code should have high CodeBLEU, got {score}"

    def test_different_code(self, mc):
        pred = "<div><p>Hello</p></div>"
        ref = "<svg><circle cx='50' cy='50' r='40'/></svg>"
        score = mc.compute_codebleu(pred, ref)
        assert score < 0.5, f"Very different code should have low CodeBLEU, got {score}"

    def test_xml_lang(self, mc):
        code = "<svg><rect width='100' height='100'/></svg>"
        score = mc.compute_codebleu(code, code, lang="xml")
        assert score > 0.9


class TestPassRate:
    def test_all_pass(self, mc):
        results = [
            {"ssim": 0.98, "error_type": "color"},
            {"ssim": 0.97, "error_type": "layout"},
        ]
        pr = mc.compute_pass_rate(results, threshold=0.95)
        assert pr["overall"] == 1.0

    def test_none_pass(self, mc):
        results = [
            {"ssim": 0.5, "error_type": "color"},
            {"ssim": 0.3, "error_type": "layout"},
        ]
        pr = mc.compute_pass_rate(results, threshold=0.95)
        assert pr["overall"] == 0.0

    def test_partial_pass(self, mc):
        results = [
            {"ssim": 0.98, "error_type": "color"},
            {"ssim": 0.80, "error_type": "color"},
            {"ssim": 0.96, "error_type": "layout"},
        ]
        pr = mc.compute_pass_rate(results, threshold=0.95)
        assert abs(pr["overall"] - 2.0 / 3.0) < 1e-6
        assert pr["per_type"]["color"] == 0.5
        assert pr["per_type"]["layout"] == 1.0

    def test_empty_results(self, mc):
        pr = mc.compute_pass_rate([])
        assert pr["overall"] == 0.0
        assert pr["per_type"] == {}


class TestEvaluateRefinement:
    def test_basic_structure(self, mc):
        """Test evaluate_refinement returns correct structure.
        Mock CLIP to avoid model download.
        """
        mc._clip_failed = True  # Skip CLIP download
        img = _make_image((128, 128, 128))
        code = "<div>test</div>"
        result = mc.evaluate_refinement(
            pred_imgs=[img],
            target_imgs=[img],
            pred_codes=[code],
            ref_codes=[code],
            error_types=["color"],
        )
        assert "ssim_mean" in result
        assert "clip_mean" in result
        assert "codebleu_mean" in result
        assert "pass_rate" in result
        assert "per_sample" in result
        assert len(result["per_sample"]) == 1
        assert result["ssim_mean"] > 0.9
        assert result["codebleu_mean"] > 0.9
        # CLIP should be -1.0 since we disabled it
        assert result["clip_mean"] == -1.0

    def test_multiple_samples(self, mc):
        mc._clip_failed = True
        imgs = [_make_image((128, 128, 128)), _make_image((200, 200, 200))]
        codes = ["<div>a</div>", "<div>b</div>"]
        result = mc.evaluate_refinement(
            pred_imgs=imgs,
            target_imgs=imgs,
            pred_codes=codes,
            ref_codes=codes,
            error_types=["color", "layout"],
        )
        assert len(result["per_sample"]) == 2
        assert result["pass_rate"]["overall"] == 1.0  # identical imgs have high SSIM
