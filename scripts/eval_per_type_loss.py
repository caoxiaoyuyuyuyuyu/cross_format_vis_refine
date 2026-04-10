"""D-1 Diagnostic: Per-error-type training loss distribution.

Loads exp1a_v5 best checkpoint, runs forward pass on stratified sample
of training set, reports loss broken down by error_type.

Usage:
  python scripts/eval_per_type_loss.py \
    --data_dir data/5k_svg \
    --checkpoint_dir outputs/exp1a_v5_baseline_5k_svg/best_model/lora \
    --model_name /root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct \
    --output_json outputs/exp1a_v5_baseline_5k_svg/per_type_loss.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def stratified_sample(dataset, n_total=1000, min_per_type=100, seed=42):
    """Stratified sample: at least min_per_type per error_type, rest proportional."""
    rng = np.random.RandomState(seed)
    # Group indices by error_type
    type_indices = defaultdict(list)
    for i in range(len(dataset)):
        # Access underlying dataset for Subset
        if hasattr(dataset, 'dataset'):
            real_idx = dataset.indices[i]
            et = dataset.dataset.samples[real_idx].get("error_type", "unknown")
        else:
            et = dataset.samples[i].get("error_type", "unknown")
        type_indices[et].append(i)

    selected = []
    n_types = len(type_indices)
    guaranteed = min(min_per_type, n_total // n_types)

    # Phase 1: guaranteed minimum per type
    for et, idxs in type_indices.items():
        chosen = rng.choice(idxs, size=min(guaranteed, len(idxs)), replace=False).tolist()
        selected.extend(chosen)

    # Phase 2: fill remaining proportionally
    remaining = n_total - len(selected)
    if remaining > 0:
        selected_set = set(selected)
        pool = [i for i in range(len(dataset)) if i not in selected_set]
        if pool:
            extra = rng.choice(pool, size=min(remaining, len(pool)), replace=False).tolist()
            selected.extend(extra)

    return selected


def main(args):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("Loading base model...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, args.checkpoint_dir)
    model = model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_name)

    # Load dataset (same settings as training: hard mode)
    from scripts.train import RefinementDataset, collate_baseline
    # Use no_error_description=True for "hard" mode (compatible with all versions)
    dataset = RefinementDataset(
        args.data_dir, processor, max_length=args.max_length,
        no_error_description=True,
    )

    # Reproduce train/val split (same seed=42, val_ratio=0.2)
    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_dataset, _ = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train split: {len(train_dataset)} samples")

    # Stratified sample
    sample_indices = stratified_sample(train_dataset, n_total=args.n_samples,
                                       min_per_type=args.min_per_type, seed=42)
    subset = Subset(train_dataset, sample_indices)
    print(f"Sampled {len(subset)} for evaluation")

    collate = lambda batch: collate_baseline(batch, processor, args.max_length)
    loader = DataLoader(subset, batch_size=1, shuffle=False,
                        collate_fn=collate, num_workers=0)

    # Forward pass: collect per-sample loss
    results = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            grid_thw = batch["image_grid_thw"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pixel_values=pixel_values,
                image_grid_thw=grid_thw,
            )
            loss_val = outputs.loss.item()
            error_type = batch["error_types"][0]

            results.append({"error_type": error_type, "loss": loss_val})

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(loader)}] last_loss={loss_val:.4f} type={error_type}")

    # Aggregate by error_type
    type_losses = defaultdict(list)
    for r in results:
        type_losses[r["error_type"]].append(r["loss"])

    summary = {}
    for et, losses in sorted(type_losses.items()):
        arr = np.array(losses)
        summary[et] = {
            "n": len(arr),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
        }

    # Print table
    print("\n" + "=" * 80)
    print(f"{'error_type':<12} {'n':>5} {'mean':>8} {'std':>8} {'median':>8} {'p25':>8} {'p75':>8} {'p95':>8}")
    print("-" * 80)
    for et, s in sorted(summary.items()):
        print(f"{et:<12} {s['n']:>5} {s['mean']:>8.4f} {s['std']:>8.4f} {s['median']:>8.4f} "
              f"{s['p25']:>8.4f} {s['p75']:>8.4f} {s['p95']:>8.4f}")
    print("=" * 80)

    # Save
    output = {"summary": summary, "raw": results}
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/root/cross_format_vis_refine/data/5k_svg")
    parser.add_argument("--checkpoint_dir", default="/root/cross_format_vis_refine/outputs/exp1a_v5_baseline_5k_svg/best_model/lora")
    parser.add_argument("--model_name", default="/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--min_per_type", type=int, default=100)
    parser.add_argument("--output_json", default="/root/cross_format_vis_refine/outputs/exp1a_v5_baseline_5k_svg/per_type_loss.json")
    args = parser.parse_args()
    main(args)
