"""Generate pilot data for Exp-0 (1K SVG + 1K HTML).

Wraps SVGPipeline and HTMLPipeline to produce data in the format
expected by scripts/train.py's RefinementDataset.

Usage:
  # Full pilot (1K SVG + 1K HTML, uses HuggingFace datasets)
  python scripts/generate_pilot.py

  # SVG only, synthetic data (no HF download)
  python scripts/generate_pilot.py --svg_only --no_hf

  # Small test run
  python scripts/generate_pilot.py --num_svg 10 --num_html 10 --no_hf
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.svg_pipeline import SVGPipeline
from src.data.html_pipeline import HTMLPipeline


def normalize_metadata(metadata: list, format_type: str) -> list:
    """Normalize metadata keys so RefinementDataset can read them uniformly.

    train.py looks for: perturbed_code/perturbed_svg/perturbed_html,
    original_code/original_svg/original_html, error_type, error_description.
    We add 'original_code' and 'perturbed_code' as canonical keys.
    """
    for entry in metadata:
        if format_type == "svg":
            entry["original_code"] = entry.get("original_svg", "")
            entry["perturbed_code"] = entry.get("perturbed_svg", "")
        elif format_type == "html":
            entry["original_code"] = entry.get("original_html", "")
            entry["perturbed_code"] = entry.get("perturbed_html", "")
    return metadata


def generate_svg(num_samples: int, output_dir: Path, seed: int, use_hf: bool):
    print(f"\n{'='*60}")
    print(f"Generating {num_samples} SVG samples -> {output_dir}")
    print(f"{'='*60}")
    pipeline = SVGPipeline(render_width=256, render_height=256, seed=seed)
    pipeline.run(num_samples=num_samples, output_dir=str(output_dir), use_hf=use_hf)

    # Normalize metadata
    meta_path = output_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
        metadata = normalize_metadata(metadata, "svg")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Normalized {len(metadata)} SVG metadata entries")


def generate_html(num_samples: int, output_dir: Path, seed: int, use_hf: bool):
    print(f"\n{'='*60}")
    print(f"Generating {num_samples} HTML samples -> {output_dir}")
    print(f"{'='*60}")
    pipeline = HTMLPipeline(render_width=1280, render_height=720, seed=seed)
    pipeline.run(num_samples=num_samples, output_dir=str(output_dir), use_hf=use_hf)

    # Normalize metadata
    meta_path = output_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
        metadata = normalize_metadata(metadata, "html")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Normalized {len(metadata)} HTML metadata entries")


def main():
    parser = argparse.ArgumentParser(description="Generate pilot data for Exp-0")
    parser.add_argument("--num_svg", type=int, default=1000)
    parser.add_argument("--num_html", type=int, default=1000)
    parser.add_argument("--output_base", type=str, default="data/pilot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_hf", action="store_true", help="Use synthetic data only")
    parser.add_argument("--svg_only", action="store_true")
    parser.add_argument("--html_only", action="store_true")
    args = parser.parse_args()

    base = PROJECT_ROOT / args.output_base
    use_hf = not args.no_hf

    if not args.html_only and args.num_svg > 0:
        generate_svg(args.num_svg, base / "svg", args.seed, use_hf)

    if not args.svg_only and args.num_html > 0:
        generate_html(args.num_html, base / "html", args.seed + 1, use_hf)

    # Print summary
    print(f"\n{'='*60}")
    print("Pilot data generation complete.")
    for fmt in ["svg", "html"]:
        meta = base / fmt / "metadata.json"
        if meta.exists():
            with open(meta) as f:
                count = len(json.load(f))
            print(f"  {fmt.upper()}: {count} samples in {base / fmt}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
