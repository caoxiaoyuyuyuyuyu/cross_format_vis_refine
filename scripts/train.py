"""Exp-0 training script for DiffCode pilot experiments.

Supports two modes:
  - Baseline: Qwen2.5-VL + LoRA (no DPA, implicit comparison)
  - DiffCode: Qwen2.5-VL + LoRA + DPA (explicit diff tokens)

Usage:
  # Baseline
  python scripts/train.py --data_dir data/pilot/svg --output_dir outputs/exp0a_baseline

  # DiffCode
  python scripts/train.py --enable_dpa --data_dir data/pilot/svg --output_dir outputs/exp0b_diffcode

  # Dry-run (1 step only)
  python scripts/train.py --enable_dpa --dry_run
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


TYPE_SPECIFIC_HINTS = {
    "element": "Focus on DOM structure: check for missing, extra, or reordered elements.",
    "position": "Focus on element positioning: check coordinates, margins, padding, and offsets.",
    "size": "Focus on element dimensions: check width, height, and scaling.",
    "color": "Focus on color values: check hex codes, RGB, and opacity.",
    "style": "Focus on CSS styling: check font properties, borders, shadows, and decorations.",
    "text": "Focus on text content: check for typos, missing text, or wrong characters.",
}

# Prompt modes for ablation:
#   super-hard: no error info at all
#   hard: error type label only (current baseline)
#   hard+hints: error type + type-specific fixing hint
#   easy: error type + detailed error description
PROMPT_MODES = ("super-hard", "hard", "hard+hints", "easy")


class RefinementDataset(Dataset):
    """Dataset for visual code refinement training.

    Loads metadata.json with original/perturbed code pairs and images.
    """

    def __init__(self, data_dir: str, processor, max_length: int = 1024,
                 no_error_description: bool = False, prompt_mode: str = None):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.max_length = max_length
        # prompt_mode takes precedence; fall back to legacy flag
        if prompt_mode:
            self.prompt_mode = prompt_mode
        elif no_error_description:
            self.prompt_mode = "hard"
        else:
            self.prompt_mode = "easy"

        metadata_path = self.data_dir / "metadata.json"
        with open(metadata_path) as f:
            self.samples = json.load(f)

        self.original_imgs_dir = self.data_dir / "original_imgs"
        self.perturbed_imgs_dir = self.data_dir / "perturbed_imgs"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load images
        img_name = f"{idx:05d}.png"
        target_img_path = self.original_imgs_dir / img_name
        rendered_img_path = self.perturbed_imgs_dir / img_name

        from PIL import Image
        target_img = Image.open(target_img_path).convert("RGB")
        rendered_img = Image.open(rendered_img_path).convert("RGB")

        # Build prompt and target
        perturbed_code = sample.get("perturbed_code", sample.get("perturbed_svg", sample.get("perturbed_html", "")))
        original_code = sample.get("original_code", sample.get("original_svg", sample.get("original_html", "")))
        error_type = sample.get("error_type", "unknown")
        error_desc = sample.get("error_description", "")

        prompt = "The rendered image does not match the target."
        if self.prompt_mode == "super-hard":
            pass  # no error info
        elif self.prompt_mode == "hard":
            prompt += f" Error type: {error_type}."
        elif self.prompt_mode == "hard+hints":
            hint = TYPE_SPECIFIC_HINTS.get(error_type, "")
            prompt += f" Error type: {error_type}. {hint}"
        else:  # easy
            prompt += f" Error type: {error_type}."
            if error_desc:
                prompt += f" {error_desc}"
        prompt += (
            f"\nCurrent code:\n```\n{perturbed_code}\n```\n"
            f"Fix the code to match the target image. Output only the corrected code."
        )

        return {
            "target_img": target_img,
            "rendered_img": rendered_img,
            "prompt": prompt,
            "target_code": original_code,
            "error_type": error_type,
            "idx": idx,
        }


def collate_baseline(batch, processor, max_length=1024):
    """Collate for baseline mode: proper VL input with image tokens in input_ids."""
    from qwen_vl_utils import process_vision_info

    target_codes = [s["target_code"] for s in batch]
    error_types = [s["error_type"] for s in batch]

    # Build chat conversations with both images
    full_conversations = []
    prompt_conversations = []
    for sample in batch:
        full_conv = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["target_img"]},
                    {"type": "image", "image": sample["rendered_img"]},
                    {"type": "text", "text": sample["prompt"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"```\n{sample['target_code']}\n```"},
                ],
            },
        ]
        prompt_conv = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["target_img"]},
                    {"type": "image", "image": sample["rendered_img"]},
                    {"type": "text", "text": sample["prompt"]},
                ],
            },
        ]
        full_conversations.append(full_conv)
        prompt_conversations.append(prompt_conv)

    # Apply chat template to get text with image placeholders
    full_texts = [
        processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        for conv in full_conversations
    ]
    prompt_texts = [
        processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        for conv in prompt_conversations
    ]

    # Extract image inputs (flat list, 2 images per sample)
    all_images = []
    for conv in full_conversations:
        imgs, _ = process_vision_info(conv)
        if imgs:
            all_images.extend(imgs)

    # Process through processor (input_ids will contain image tokens)
    inputs = processor(
        text=full_texts,
        images=all_images if all_images else None,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    # Process prompt-only to get prompt token counts for label masking
    prompt_images = []
    for conv in prompt_conversations:
        imgs, _ = process_vision_info(conv)
        if imgs:
            prompt_images.extend(imgs)

    prompt_inputs = processor(
        text=prompt_texts,
        images=prompt_images if prompt_images else None,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    # Build labels: mask prompt tokens with -100
    labels = inputs["input_ids"].clone()
    for i in range(len(batch)):
        prompt_len = prompt_inputs["attention_mask"][i].sum().item()
        labels[i, :prompt_len] = -100
    labels[inputs["attention_mask"] == 0] = -100

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "pixel_values": inputs.get("pixel_values"),
        "image_grid_thw": inputs.get("image_grid_thw"),
        "labels": labels,
        "error_types": error_types,
        "target_codes": target_codes,
    }


def collate_diffcode(batch, processor, max_length=1024):
    """Collate for DiffCode mode: VL path (same as baseline) + separate DPA inputs."""
    from qwen_vl_utils import process_vision_info

    target_codes = [s["target_code"] for s in batch]
    error_types = [s["error_type"] for s in batch]

    # === VL path: same as baseline (both images in conversation) ===
    full_conversations = []
    prompt_conversations = []
    for sample in batch:
        full_conv = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["target_img"]},
                    {"type": "image", "image": sample["rendered_img"]},
                    {"type": "text", "text": sample["prompt"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"```\n{sample['target_code']}\n```"},
                ],
            },
        ]
        prompt_conv = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["target_img"]},
                    {"type": "image", "image": sample["rendered_img"]},
                    {"type": "text", "text": sample["prompt"]},
                ],
            },
        ]
        full_conversations.append(full_conv)
        prompt_conversations.append(prompt_conv)

    full_texts = [
        processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        for conv in full_conversations
    ]
    prompt_texts = [
        processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        for conv in prompt_conversations
    ]

    all_images = []
    for conv in full_conversations:
        imgs, _ = process_vision_info(conv)
        if imgs:
            all_images.extend(imgs)

    inputs = processor(
        text=full_texts,
        images=all_images if all_images else None,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    prompt_images = []
    for conv in prompt_conversations:
        imgs, _ = process_vision_info(conv)
        if imgs:
            prompt_images.extend(imgs)

    prompt_inputs = processor(
        text=prompt_texts,
        images=prompt_images if prompt_images else None,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    labels = inputs["input_ids"].clone()
    for i in range(len(batch)):
        prompt_len = prompt_inputs["attention_mask"][i].sum().item()
        labels[i, :prompt_len] = -100
    labels[inputs["attention_mask"] == 0] = -100

    # === DPA path: process target and rendered images separately ===
    target_image_inputs, _ = process_vision_info(
        [{"role": "user", "content": [{"type": "image", "image": s["target_img"]}]} for s in batch]
    )
    rendered_image_inputs, _ = process_vision_info(
        [{"role": "user", "content": [{"type": "image", "image": s["rendered_img"]}]} for s in batch]
    )

    target_processed = processor(
        images=target_image_inputs,
        text=[""] * len(batch),
        return_tensors="pt",
        padding=True,
    )
    rendered_processed = processor(
        images=rendered_image_inputs,
        text=[""] * len(batch),
        return_tensors="pt",
        padding=True,
    )

    return {
        # VL path inputs (image tokens in input_ids)
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "pixel_values": inputs.get("pixel_values"),
        "image_grid_thw": inputs.get("image_grid_thw"),
        "labels": labels,
        # DPA path inputs (separate images for diff computation)
        "target_pixel_values": target_processed.get("pixel_values"),
        "target_grid_thw": target_processed.get("image_grid_thw"),
        "rendered_pixel_values": rendered_processed.get("pixel_values"),
        "rendered_grid_thw": rendered_processed.get("image_grid_thw"),
        # Metadata
        "error_types": error_types,
        "target_codes": target_codes,
    }


def create_baseline_model(args):
    """Create baseline model: Qwen2.5-VL + LoRA only (no DPA)."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(args.model_name)

    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False

    # Apply LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


def create_diffcode_model(args):
    """Create DiffCode model: Qwen2.5-VL + LoRA + DPA."""
    from src.model.diffcode import DiffCodeModel

    model = DiffCodeModel(
        model_name=args.model_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        hook_layers=(7, 23, 31),
    )
    param_counts = model.get_trainable_param_count()
    print(f"Trainable params: DPA={param_counts['dpa']:,}, LoRA={param_counts['lora']:,}, Total={param_counts['total']:,}")

    return model, model.processor


def train_baseline_step(model, batch, device):
    """Single training step for baseline model."""
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
    return outputs.loss


def train_diffcode_step(model, batch, device):
    """Single training step for DiffCode model."""
    kwargs = dict(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["labels"].to(device),
        # VL path: image tokens in conversation (same as baseline)
        pixel_values=batch["pixel_values"].to(device, dtype=torch.bfloat16) if batch.get("pixel_values") is not None else None,
        image_grid_thw=batch["image_grid_thw"].to(device) if batch.get("image_grid_thw") is not None else None,
        # DPA path: separate target/rendered for diff tokens
        target_pixel_values=batch["target_pixel_values"].to(device, dtype=torch.bfloat16) if batch.get("target_pixel_values") is not None else None,
        target_grid_thw=batch["target_grid_thw"].to(device) if batch.get("target_grid_thw") is not None else None,
        rendered_pixel_values=batch["rendered_pixel_values"].to(device, dtype=torch.bfloat16) if batch.get("rendered_pixel_values") is not None else None,
        rendered_grid_thw=batch["rendered_grid_thw"].to(device) if batch.get("rendered_grid_thw") is not None else None,
    )
    outputs = model(**kwargs)
    return outputs["loss"]


def train(args):
    """Main training loop."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize wandb
    if not args.dry_run and args.wandb_project:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"exp0_{'diffcode' if args.enable_dpa else 'baseline'}",
            config=vars(args),
        )

    # Create model
    if args.enable_dpa:
        print("Creating DiffCode model (LoRA + DPA)...")
        model, processor = create_diffcode_model(args)
    else:
        print("Creating baseline model (LoRA only)...")
        model, processor = create_baseline_model(args)

    model = model.to(device)

    # Create dataset
    print(f"Loading data from {args.data_dir}...")
    dataset = RefinementDataset(args.data_dir, processor, max_length=args.max_length,
                                no_error_description=args.no_error_description,
                                prompt_mode=args.prompt_mode)
    print(f"Total samples: {len(dataset)}")

    # Template-level train/val split (prevents data leakage from shared templates)
    import hashlib
    import random as _random
    from collections import defaultdict

    template_groups = defaultdict(list)
    for i in range(len(dataset)):
        sample = dataset.samples[i]
        original = sample.get('original_svg') or sample.get('original_html') or sample.get('original_code')
        h = hashlib.md5(original.encode()).hexdigest()
        template_groups[h].append(i)

    rng = _random.Random(args.seed)
    templates = sorted(template_groups.keys())
    rng.shuffle(templates)
    n_val_templates = max(1, int(len(templates) * args.val_ratio))
    val_templates = set(templates[:n_val_templates])
    train_templates = set(templates[n_val_templates:])

    val_indices = [i for t in val_templates for i in template_groups[t]]
    train_indices = [i for t in train_templates for i in template_groups[t]]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    print(f"Template-level split: {len(train_templates)} train templates ({len(train_indices)} samples) "
          f"/ {len(val_templates)} val templates ({len(val_indices)} samples)")
    print(f"Template overlap (must be 0): {len(train_templates & val_templates)}")

    if args.enable_dpa:
        collate = lambda batch: collate_diffcode(batch, processor, args.max_length)
    else:
        collate = lambda batch: collate_baseline(batch, processor, args.max_length)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=0, pin_memory=True,
    )

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # LR scheduler with warmup
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step_fn = train_diffcode_step if args.enable_dpa else train_baseline_step

    # Dry-run: just 1 step
    if args.dry_run:
        print("\n=== DRY RUN: 1 step ===")
        model.train()
        batch = next(iter(train_loader))
        loss = step_fn(model, batch, device)
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item():.4f}")
        print("DRY_RUN_OK")
        return

    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss = step_fn(model, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % args.log_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}/{args.num_epochs} Step {batch_idx+1}/{len(train_loader)} "
                      f"Loss={loss.item():.4f} AvgLoss={avg_loss:.4f} LR={lr:.2e}")

                if args.wandb_project:
                    import wandb
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/avg_loss": avg_loss,
                        "train/lr": lr,
                        "train/epoch": epoch + 1,
                        "train/global_step": global_step,
                    })

        epoch_time = time.time() - epoch_start
        train_avg_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                loss = step_fn(model, batch, device)
                val_loss += loss.item()
        val_avg_loss = val_loss / max(1, len(val_loader))

        print(f"\nEpoch {epoch+1}/{args.num_epochs}: "
              f"TrainLoss={train_avg_loss:.4f} ValLoss={val_avg_loss:.4f} Time={epoch_time:.0f}s")

        if args.wandb_project:
            import wandb
            wandb.log({
                "val/loss": val_avg_loss,
                "val/epoch": epoch + 1,
            })

        # Save best model
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            save_path = Path(args.output_dir) / "best_model"
            os.makedirs(save_path, exist_ok=True)
            if args.enable_dpa:
                # Save DPA + LoRA separately
                torch.save(model.dpa.state_dict(), save_path / "dpa.pt")
                model.base_model.save_pretrained(save_path / "lora")
            else:
                model.save_pretrained(save_path / "lora")
            print(f"  Saved best model (val_loss={val_avg_loss:.4f})")

    # Save final model
    save_path = Path(args.output_dir) / "final_model"
    os.makedirs(save_path, exist_ok=True)
    if args.enable_dpa:
        torch.save(model.dpa.state_dict(), save_path / "dpa.pt")
        model.base_model.save_pretrained(save_path / "lora")
    else:
        model.save_pretrained(save_path / "lora")

    # Save training config
    with open(Path(args.output_dir) / "train_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Outputs saved to: {args.output_dir}")

    if args.wandb_project:
        import wandb
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="DiffCode Exp-0 Training")

    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--enable_dpa", action="store_true", help="Enable DPA (DiffCode mode)")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)

    # Data
    parser.add_argument("--data_dir", type=str, default="data/pilot/svg")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--val_ratio", type=float, default=0.2)

    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=10)

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/exp0")

    # W&B
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")

    # Prompt
    parser.add_argument("--no_error_description", action="store_true",
                        help="Remove error_description from prompt (hard condition)")
    parser.add_argument("--prompt_mode", type=str, default=None,
                        choices=["super-hard", "hard", "hard+hints", "easy"],
                        help="Prompt ablation mode (overrides --no_error_description)")

    # Dry-run
    parser.add_argument("--dry_run", action="store_true", help="Run 1 step only to verify pipeline")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train(args)
