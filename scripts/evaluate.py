"""Evaluate a trained model on the validation set.

Loads the best_model checkpoint, runs inference on val samples,
renders the generated code, and computes SSIM + CLIP-Score.

Usage:
  # Baseline
  python scripts/evaluate.py \
    --model_name /root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct \
    --checkpoint_dir outputs/exp0a_baseline_svg/best_model \
    --data_dir data/pilot/svg

  # DiffCode
  python scripts/evaluate.py \
    --model_name /root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct \
    --checkpoint_dir outputs/exp0b_diffcode_svg/best_model \
    --data_dir data/pilot/svg --enable_dpa
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import RefinementDataset


def load_baseline_model(args):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(args.model_name)

    lora_path = Path(args.checkpoint_dir) / "lora"
    model = PeftModel.from_pretrained(base_model, str(lora_path))
    model.eval()
    return model, processor


def load_diffcode_model(args):
    """Load DiffCode model with LoRA + DPA for inference."""
    from src.model.diffcode import DiffCodeModel
    from peft import PeftModel

    model = DiffCodeModel(
        model_name=args.model_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        hook_layers=(7, 23, 31),
    )

    # Load trained LoRA weights on top of the freshly initialized LoRA
    lora_path = Path(args.checkpoint_dir) / "lora"
    if lora_path.exists():
        # Merge fresh LoRA then load trained adapter
        model.base_model = model.base_model.merge_and_unload()
        model.base_model = PeftModel.from_pretrained(
            model.base_model, str(lora_path)
        )
        # Re-register hooks (merge_and_unload may reset them)
        model.register_vit_hooks()

    # Load DPA weights
    dpa_path = Path(args.checkpoint_dir) / "dpa.pt"
    if dpa_path.exists():
        model.dpa.load_state_dict(torch.load(dpa_path, map_location="cpu"))
        print(f"Loaded DPA weights from {dpa_path}")

    model.eval()
    return model, model.processor


def render_svg_to_image(svg_code: str, width: int = 256, height: int = 256):
    """Render SVG code to PIL Image."""
    from PIL import Image
    import io

    try:
        import cairosvg
        png_data = cairosvg.svg2png(
            bytestring=svg_code.encode("utf-8"),
            output_width=width,
            output_height=height,
        )
        return Image.open(io.BytesIO(png_data)).convert("RGB")
    except Exception:
        # Fallback: return blank image
        return Image.new("RGB", (width, height), (255, 255, 255))


def render_html_to_image(html_code: str, width: int = 1280, height: int = 720):
    """Render HTML code to PIL Image via wkhtmltoimage."""
    from PIL import Image
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = os.path.join(tmpdir, "input.html")
        img_path = os.path.join(tmpdir, "shot.png")
        with open(html_path, "w") as f:
            f.write(html_code)
        try:
            subprocess.run(
                [
                    "wkhtmltoimage", "--quiet",
                    "--width", str(width), "--height", str(height),
                    "--quality", "90",
                    "--load-error-handling", "ignore",
                    "--load-media-error-handling", "ignore",
                    "--disable-javascript",
                    "--enable-local-file-access",
                    html_path, img_path,
                ],
                check=False, timeout=30,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            return Image.new("RGB", (width, height), (255, 255, 255))
        if not os.path.exists(img_path):
            return Image.new("RGB", (width, height), (255, 255, 255))
        return Image.open(img_path).convert("RGB").copy()


def extract_code_from_response(response: str) -> str:
    """Extract code from model response (handles ```...``` blocks)."""
    # Try to find code block
    match = re.search(r"```(?:\w*\n)?(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


@torch.no_grad()
def run_inference(model, processor, sample, device, enable_dpa=False, max_new_tokens=1024):
    """Run inference on a single sample and return the generated code."""
    from qwen_vl_utils import process_vision_info

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["target_img"]},
                {"type": "image", "image": sample["rendered_img"]},
                {"type": "text", "text": sample["prompt"]},
            ],
        },
    ]

    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    images, _ = process_vision_info(conversation)

    inputs = processor(
        text=[text],
        images=images if images else None,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    if enable_dpa:
        # DPA-augmented generation: process target/rendered images separately for DPA
        target_conv = [{"role": "user", "content": [{"type": "image", "image": sample["target_img"]}]}]
        rendered_conv = [{"role": "user", "content": [{"type": "image", "image": sample["rendered_img"]}]}]
        target_imgs, _ = process_vision_info(target_conv)
        rendered_imgs, _ = process_vision_info(rendered_conv)

        target_processed = processor(
            text=["placeholder"], images=target_imgs, return_tensors="pt", padding=True,
        )
        rendered_processed = processor(
            text=["placeholder"], images=rendered_imgs, return_tensors="pt", padding=True,
        )

        generated = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values", torch.empty(0)).to(device, dtype=torch.bfloat16),
            image_grid_thw=inputs.get("image_grid_thw", torch.empty(0)).to(device),
            target_pixel_values=target_processed["pixel_values"].to(device, dtype=torch.bfloat16),
            target_grid_thw=target_processed["image_grid_thw"].to(device),
            rendered_pixel_values=rendered_processed["pixel_values"].to(device, dtype=torch.bfloat16),
            rendered_grid_thw=rendered_processed["image_grid_thw"].to(device),
            max_new_tokens=max_new_tokens,
        )
        # generate() returns only generated tokens (no input prefix)
        response = processor.decode(generated[0], skip_special_tokens=True)
    else:
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        response = processor.decode(generated, skip_special_tokens=True)

    return extract_code_from_response(response)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on val set")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--enable_dpa", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=0, help="Limit eval samples (0=all)")
    parser.add_argument("--no_error_description", action="store_true", help="Hard condition: no error description in prompt")
    parser.add_argument("--prompt_mode", type=str, default=None,
                        choices=["super-hard", "hard", "hard+hints", "easy"],
                        help="Prompt ablation mode (overrides --no_error_description)")
    parser.add_argument("--format", type=str, default="svg", choices=["svg", "html"],
                        help="Code format for rendering (svg or html)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("Loading model...")
    if args.enable_dpa:
        model, processor = load_diffcode_model(args)
    else:
        model, processor = load_baseline_model(args)
    model = model.to(device)

    # Load dataset and get val split (same split as training)
    dataset = RefinementDataset(args.data_dir, processor,
                                no_error_description=args.no_error_description,
                                prompt_mode=args.prompt_mode)
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    _, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"Val samples: {len(val_dataset)}")

    if args.max_samples > 0:
        indices = list(range(min(args.max_samples, len(val_dataset))))
    else:
        indices = list(range(len(val_dataset)))

    # Initialize metrics
    from src.evaluation.metrics import MetricsComputer
    metrics = MetricsComputer(device=str(device))

    results = []
    t0 = time.time()

    for i, vi in enumerate(indices):
        sample = val_dataset[vi]
        print(f"\r[{i+1}/{len(indices)}] Evaluating...", end="", flush=True)

        # Generate refined code
        pred_code = run_inference(
            model, processor, sample, device,
            enable_dpa=args.enable_dpa,
            max_new_tokens=args.max_new_tokens,
        )

        # Render predicted code
        if args.format == "html":
            pred_img = render_html_to_image(pred_code)
        else:
            pred_img = render_svg_to_image(pred_code)
        target_img = sample["target_img"]

        # Compute metrics
        ssim = metrics.compute_ssim(pred_img, target_img)
        clip_score = metrics.compute_clip_score(pred_img, target_img)

        results.append({
            "idx": sample["idx"],
            "error_type": sample["error_type"],
            "ssim": ssim,
            "clip_score": clip_score,
            "pred_code": pred_code,
            "gt_code": sample.get("target_code", ""),
        })

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")

    # Aggregate
    ssim_scores = [r["ssim"] for r in results]
    clip_scores = [r["clip_score"] for r in results if r["clip_score"] >= 0]

    summary = {
        "n_samples": len(results),
        "ssim_mean": float(np.mean(ssim_scores)),
        "ssim_std": float(np.std(ssim_scores)),
        "clip_mean": float(np.mean(clip_scores)) if clip_scores else -1.0,
        "clip_std": float(np.std(clip_scores)) if clip_scores else 0.0,
        "pass_rate_095": sum(1 for s in ssim_scores if s > 0.95) / len(ssim_scores),
        "elapsed_seconds": elapsed,
    }

    # Per error type breakdown
    from collections import defaultdict
    type_scores = defaultdict(list)
    for r in results:
        type_scores[r["error_type"]].append(r["ssim"])
    summary["per_error_type"] = {
        et: {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "n": len(scores)}
        for et, scores in type_scores.items()
    }

    print("\n" + "=" * 60)
    print(f"SSIM:       {summary['ssim_mean']:.4f} ± {summary['ssim_std']:.4f}")
    print(f"CLIP-Score: {summary['clip_mean']:.4f} ± {summary['clip_std']:.4f}")
    print(f"Pass Rate:  {summary['pass_rate_095']:.1%} (SSIM > 0.95)")
    print("Per error type:")
    for et, stats in summary["per_error_type"].items():
        print(f"  {et}: SSIM={stats['mean']:.4f}±{stats['std']:.4f} (n={stats['n']})")
    print("=" * 60)

    # Save results
    output_file = args.output_file or str(Path(args.checkpoint_dir).parent / "eval_results.json")
    output = {"summary": summary, "per_sample": results}
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
