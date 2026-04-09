"""Evaluation metrics for cross-format visual code refinement."""

import os
import warnings
import numpy as np
from PIL import Image
from typing import List, Optional


class MetricsComputer:
    def __init__(self, device="cpu"):
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._clip_failed = False
        self.device = device

    def _to_numpy(self, img) -> np.ndarray:
        """Convert PIL.Image or numpy array to numpy array."""
        if isinstance(img, Image.Image):
            return np.array(img)
        return np.asarray(img)

    def _resize_to_match(self, img_a: np.ndarray, img_b: np.ndarray):
        """Resize img_b to match img_a's dimensions if they differ."""
        if img_a.shape[:2] != img_b.shape[:2]:
            target_size = (img_a.shape[1], img_a.shape[0])  # PIL uses (w, h)
            img_b_pil = Image.fromarray(img_b).resize(target_size, Image.LANCZOS)
            img_b = np.array(img_b_pil)
        return img_a, img_b

    def compute_ssim(self, img_a, img_b) -> float:
        """Compute SSIM between two images.

        Input: PIL.Image or numpy array.
        Returns 0-1 float.
        """
        from skimage.metrics import structural_similarity as ssim
        from skimage.color import rgb2gray

        img_a = self._to_numpy(img_a)
        img_b = self._to_numpy(img_b)
        img_a, img_b = self._resize_to_match(img_a, img_b)

        # Convert to grayscale if RGB/RGBA
        if img_a.ndim == 3:
            if img_a.shape[2] == 4:
                img_a = img_a[:, :, :3]
            img_a = rgb2gray(img_a)
        if img_b.ndim == 3:
            if img_b.shape[2] == 4:
                img_b = img_b[:, :, :3]
            img_b = rgb2gray(img_b)

        score = ssim(img_a, img_b, data_range=1.0)
        return float(score)

    def _load_clip(self):
        """Lazy load CLIP model."""
        if self._clip_model is not None or self._clip_failed:
            return
        try:
            import open_clip
            import torch

            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='openai'
            )
            model = model.to(self.device).eval()
            self._clip_model = model
            self._clip_preprocess = preprocess
        except Exception as e:
            warnings.warn(f"Failed to load CLIP model: {e}")
            self._clip_failed = True
            self._clip_model = None

    def compute_clip_score(self, img_a, img_b) -> float:
        """Compute CLIP image similarity between two images.

        Returns cosine similarity 0-1 float. Returns -1.0 on failure.
        """
        try:
            import torch
            self._load_clip()
            if self._clip_model is None:
                return -1.0

            if not isinstance(img_a, Image.Image):
                img_a = Image.fromarray(np.asarray(img_a))
            if not isinstance(img_b, Image.Image):
                img_b = Image.fromarray(np.asarray(img_b))

            # Ensure RGB
            img_a = img_a.convert("RGB")
            img_b = img_b.convert("RGB")

            with torch.no_grad():
                feat_a = self._clip_model.encode_image(
                    self._clip_preprocess(img_a).unsqueeze(0).to(self.device)
                )
                feat_b = self._clip_model.encode_image(
                    self._clip_preprocess(img_b).unsqueeze(0).to(self.device)
                )
                # Normalize
                feat_a = feat_a / feat_a.norm(dim=-1, keepdim=True)
                feat_b = feat_b / feat_b.norm(dim=-1, keepdim=True)
                similarity = (feat_a @ feat_b.T).item()

            # Clamp to [0, 1]
            return float(max(0.0, min(1.0, similarity)))
        except Exception as e:
            warnings.warn(f"CLIP score computation failed: {e}")
            return -1.0

    def compute_codebleu(self, pred: str, ref: str, lang: str = "html") -> float:
        """Compute CodeBLEU score.

        lang: "html" or "xml" (SVG uses xml).
        Falls back to sacrebleu if lang is unsupported.
        Returns 0-1 float.
        """
        # Map SVG to xml
        lang_map = {"svg": "xml"}
        lang = lang_map.get(lang, lang)

        try:
            from codebleu import calc_codebleu
            # codebleu supported languages
            supported = {"java", "javascript", "c_sharp", "php", "c", "cpp", "python", "go", "ruby", "rust"}

            effective_lang = lang
            if lang not in supported:
                # html/xml -> try javascript as closest fallback
                effective_lang = "javascript"

            try:
                result = calc_codebleu([ref], [pred], lang=effective_lang)
                return float(result["codebleu"])
            except Exception:
                # If codebleu fails even with fallback lang, use sacrebleu
                pass
        except ImportError:
            pass

        # Fallback to sacrebleu
        try:
            import sacrebleu
            bleu = sacrebleu.corpus_bleu([pred], [[ref]])
            return float(bleu.score / 100.0)  # sacrebleu returns 0-100
        except Exception:
            warnings.warn("Both codebleu and sacrebleu failed")
            return 0.0

    def compute_pass_rate(self, results: list, threshold: float = 0.95) -> dict:
        """Compute pass rate from a list of results.

        Input: list of {"ssim": float, "error_type": str}
        Returns: {"overall": float, "per_type": {"color": float, ...}}
        """
        if not results:
            return {"overall": 0.0, "per_type": {}}

        total_pass = sum(1 for r in results if r["ssim"] > threshold)
        overall = total_pass / len(results)

        # Group by error_type
        type_groups = {}
        for r in results:
            et = r["error_type"]
            if et not in type_groups:
                type_groups[et] = {"total": 0, "passed": 0}
            type_groups[et]["total"] += 1
            if r["ssim"] > threshold:
                type_groups[et]["passed"] += 1

        per_type = {
            et: g["passed"] / g["total"] for et, g in type_groups.items()
        }

        return {"overall": overall, "per_type": per_type}

    def evaluate_refinement(
        self,
        pred_imgs: list,
        target_imgs: list,
        pred_codes: list,
        ref_codes: list,
        error_types: list,
    ) -> dict:
        """One-stop evaluation.

        Returns: {
            "ssim_mean": float, "clip_mean": float, "codebleu_mean": float,
            "pass_rate": {"overall": float, "per_type": {...}},
            "per_sample": [{"ssim": float, "clip": float, "codebleu": float}, ...]
        }
        """
        per_sample = []
        ssim_scores = []
        clip_scores = []
        codebleu_scores = []

        for i in range(len(pred_imgs)):
            s = self.compute_ssim(pred_imgs[i], target_imgs[i])
            c = self.compute_clip_score(pred_imgs[i], target_imgs[i])
            cb = self.compute_codebleu(pred_codes[i], ref_codes[i])

            per_sample.append({"ssim": s, "clip": c, "codebleu": cb})
            ssim_scores.append(s)
            clip_scores.append(c)
            codebleu_scores.append(cb)

        # Build results for pass_rate
        pass_results = [
            {"ssim": per_sample[i]["ssim"], "error_type": error_types[i]}
            for i in range(len(per_sample))
        ]

        # Filter out failed clip scores (-1.0) for mean
        valid_clips = [c for c in clip_scores if c >= 0]

        return {
            "ssim_mean": float(np.mean(ssim_scores)) if ssim_scores else 0.0,
            "clip_mean": float(np.mean(valid_clips)) if valid_clips else -1.0,
            "codebleu_mean": float(np.mean(codebleu_scores)) if codebleu_scores else 0.0,
            "pass_rate": self.compute_pass_rate(pass_results),
            "per_sample": per_sample,
        }
