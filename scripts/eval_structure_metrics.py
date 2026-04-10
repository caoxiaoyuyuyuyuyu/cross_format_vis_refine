"""Compute structure-aware metrics on saved predictions.

Reads predictions.json (with pred_code / gt_code) and computes:
  1. Tree Edit Distance (TED) via zss
  2. Node-level precision / recall / F1
  3. Element exact match rate

Usage:
  python scripts/eval_structure_metrics.py \
    --predictions outputs/exp1a_v5_baseline_5k_svg/predictions.json \
    --output outputs/exp1a_v5_baseline_5k_svg/structure_metrics.json
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_svg_tree(svg_code: str):
    """Parse SVG code into an lxml etree. Returns None on failure."""
    from lxml import etree

    svg_code = svg_code.strip()
    if not svg_code:
        return None
    # Wrap in svg tag if not present
    if not svg_code.startswith("<"):
        return None
    try:
        # Remove XML declaration if present
        svg_code = re.sub(r'<\?xml[^?]*\?>', '', svg_code).strip()
        root = etree.fromstring(svg_code.encode("utf-8"))
        return root
    except Exception:
        try:
            # Try wrapping in a root element
            wrapped = f"<root>{svg_code}</root>"
            root = etree.fromstring(wrapped.encode("utf-8"))
            return root
        except Exception:
            return None


def tree_to_zss(node):
    """Convert lxml element to zss Node recursively."""
    import zss

    # Label = tag name (strip namespace)
    tag = node.tag
    if "}" in tag:
        tag = tag.split("}")[-1]

    zss_node = zss.Node(tag)
    for child in node:
        child_zss = tree_to_zss(child)
        zss_node.addkid(child_zss)
    return zss_node


def count_tree_nodes(node):
    """Count total nodes in an lxml tree."""
    count = 1
    for child in node:
        count += count_tree_nodes(child)
    return count


def compute_ted(pred_tree, gt_tree):
    """Compute tree edit distance between two lxml trees."""
    import zss

    pred_zss = tree_to_zss(pred_tree)
    gt_zss = tree_to_zss(gt_tree)

    dist = zss.simple_distance(pred_zss, gt_zss)
    max_nodes = max(count_tree_nodes(pred_tree), count_tree_nodes(gt_tree))
    normalized = dist / max_nodes if max_nodes > 0 else 0.0

    return float(dist), float(normalized)


def extract_node_set(node, path=""):
    """Extract set of (path, tag, sorted_attribs) tuples from tree."""
    tag = node.tag
    if "}" in tag:
        tag = tag.split("}")[-1]

    current_path = f"{path}/{tag}"
    attribs = tuple(sorted(node.attrib.items()))
    nodes = {(current_path, tag, attribs)}

    for i, child in enumerate(node):
        child_path = f"{current_path}[{i}]"
        nodes |= extract_node_set(child, child_path)

    return nodes


def compute_node_metrics(pred_tree, gt_tree):
    """Compute node-level precision, recall, F1."""
    pred_nodes = extract_node_set(pred_tree)
    gt_nodes = extract_node_set(gt_tree)

    if not pred_nodes and not gt_nodes:
        return 1.0, 1.0, 1.0

    tp = len(pred_nodes & gt_nodes)
    precision = tp / len(pred_nodes) if pred_nodes else 0.0
    recall = tp / len(gt_nodes) if gt_nodes else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def normalize_code(code: str) -> str:
    """Normalize SVG code for exact match: collapse whitespace, sort attributes."""
    code = re.sub(r'\s+', ' ', code.strip())
    return code


def compute_exact_match(pred_code: str, gt_code: str) -> bool:
    """Check if predicted code matches ground truth after normalization."""
    return normalize_code(pred_code) == normalize_code(gt_code)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to eval_results.json with pred_code/gt_code")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    with open(args.predictions) as f:
        data = json.load(f)

    samples = data.get("per_sample", data if isinstance(data, list) else [])

    # Check that pred_code exists
    if not samples or "pred_code" not in samples[0]:
        print("ERROR: predictions file does not contain pred_code. Re-run evaluate.py first.")
        sys.exit(1)

    all_metrics = []
    parse_failures = 0

    for i, s in enumerate(samples):
        pred_code = s.get("pred_code", "")
        gt_code = s.get("gt_code", "")
        error_type = s.get("error_type", "unknown")

        pred_tree = parse_svg_tree(pred_code)
        gt_tree = parse_svg_tree(gt_code)

        entry = {
            "idx": s.get("idx", i),
            "error_type": error_type,
            "ssim": s.get("ssim", -1),
        }

        if pred_tree is not None and gt_tree is not None:
            ted, ted_norm = compute_ted(pred_tree, gt_tree)
            prec, rec, f1 = compute_node_metrics(pred_tree, gt_tree)
            exact = compute_exact_match(pred_code, gt_code)

            entry.update({
                "ted": ted,
                "ted_normalized": ted_norm,
                "node_precision": prec,
                "node_recall": rec,
                "node_f1": f1,
                "exact_match": exact,
                "parse_ok": True,
            })
        else:
            parse_failures += 1
            entry.update({
                "ted": -1, "ted_normalized": -1,
                "node_precision": 0, "node_recall": 0, "node_f1": 0,
                "exact_match": False,
                "parse_ok": False,
            })

        all_metrics.append(entry)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(samples)}")

    # Aggregate
    valid = [m for m in all_metrics if m["parse_ok"]]

    def agg(items, key):
        vals = [m[key] for m in items]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)} if vals else {"mean": -1, "std": 0, "n": 0}

    summary = {
        "n_total": len(all_metrics),
        "n_valid": len(valid),
        "parse_failures": parse_failures,
        "overall": {
            "ted": agg(valid, "ted"),
            "ted_normalized": agg(valid, "ted_normalized"),
            "node_precision": agg(valid, "node_precision"),
            "node_recall": agg(valid, "node_recall"),
            "node_f1": agg(valid, "node_f1"),
            "exact_match_rate": sum(1 for m in valid if m["exact_match"]) / len(valid) if valid else 0,
        },
    }

    # Per error type
    by_type = defaultdict(list)
    for m in all_metrics:
        by_type[m["error_type"]].append(m)

    summary["per_error_type"] = {}
    for et, items in sorted(by_type.items()):
        valid_items = [m for m in items if m["parse_ok"]]
        summary["per_error_type"][et] = {
            "n": len(items),
            "n_valid": len(valid_items),
            "ssim": agg(valid_items, "ssim"),
            "ted": agg(valid_items, "ted"),
            "ted_normalized": agg(valid_items, "ted_normalized"),
            "node_precision": agg(valid_items, "node_precision"),
            "node_recall": agg(valid_items, "node_recall"),
            "node_f1": agg(valid_items, "node_f1"),
            "exact_match_rate": sum(1 for m in valid_items if m["exact_match"]) / len(valid_items) if valid_items else 0,
        }

    # Print summary
    print("\n" + "=" * 70)
    print(f"Structure Metrics Summary  (valid: {len(valid)}/{len(all_metrics)}, parse failures: {parse_failures})")
    print("=" * 70)
    o = summary["overall"]
    print(f"  TED:            {o['ted']['mean']:.2f} ± {o['ted']['std']:.2f}")
    print(f"  TED (norm):     {o['ted_normalized']['mean']:.4f} ± {o['ted_normalized']['std']:.4f}")
    print(f"  Node Precision: {o['node_precision']['mean']:.4f}")
    print(f"  Node Recall:    {o['node_recall']['mean']:.4f}")
    print(f"  Node F1:        {o['node_f1']['mean']:.4f}")
    print(f"  Exact Match:    {o['exact_match_rate']:.1%}")
    print()
    print("Per error type:")
    print(f"  {'Type':<12} {'N':>4} {'SSIM':>8} {'TED↓':>8} {'TED_n↓':>8} {'F1↑':>8} {'EM↑':>8}")
    print(f"  {'-'*12} {'---':>4} {'------':>8} {'------':>8} {'------':>8} {'------':>8} {'------':>8}")
    for et, stats in sorted(summary["per_error_type"].items()):
        print(f"  {et:<12} {stats['n']:>4} "
              f"{stats['ssim']['mean']:>8.4f} "
              f"{stats['ted']['mean']:>8.2f} "
              f"{stats['ted_normalized']['mean']:>8.4f} "
              f"{stats['node_f1']['mean']:>8.4f} "
              f"{stats['exact_match_rate']:>8.1%}")
    print("=" * 70)

    # Save
    output_data = {"summary": summary, "per_sample": all_metrics}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
