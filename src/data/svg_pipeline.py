"""SVG perturbation pipeline for generating training pairs.

Generates (original_svg, perturbed_svg, original_img, perturbed_img,
error_type, error_description) tuples for cross-format visual refinement.
"""

import argparse
import copy
import io
import json
import os
import random
import re
import string
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import cairosvg
from PIL import Image

# Allow running as script from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.error_taxonomy import ErrorCategory, SVG_ERROR_SUBTYPES

# SVG namespace
SVG_NS = "http://www.w3.org/2000/svg"
NS_MAP = {"svg": SVG_NS}
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")

# Elements that are visible and perturbable
VISIBLE_TAGS = {
    f"{{{SVG_NS}}}{t}"
    for t in ("rect", "circle", "ellipse", "line", "polyline", "polygon", "path", "text", "image", "use")
}


# ── Rendering ──────────────────────────────────────────────────────────

def render_svg(svg_code: str, width: int = 256, height: int = 256) -> Image.Image:
    """Render SVG string to a PIL Image via CairoSVG."""
    png_bytes = cairosvg.svg2png(
        bytestring=svg_code.encode("utf-8"),
        output_width=width,
        output_height=height,
    )
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


# ── Perturbation engine ───────────────────────────────────────────────

def _random_hex_color() -> str:
    return f"#{random.randint(0, 0xFFFFFF):06x}"


def _shift_value(val_str: str, lo: int = 5, hi: int = 20) -> str:
    """Shift a numeric string by a random offset."""
    try:
        val = float(val_str)
    except (ValueError, TypeError):
        return val_str
    offset = random.choice([-1, 1]) * random.randint(lo, hi)
    return str(round(val + offset, 2))


class SVGPerturbation:
    """Implements 6 perturbation types on parsed SVG ElementTrees."""

    # ── color ──────────────────────────────────────────────────────

    @staticmethod
    def perturb_color(tree: ET.ElementTree) -> Tuple[Optional[ET.ElementTree], str]:
        """Randomly change fill or stroke color of a visible element."""
        root = tree.getroot()
        candidates: List[ET.Element] = []
        for elem in root.iter():
            if elem.tag in VISIBLE_TAGS:
                if elem.get("fill") or elem.get("stroke"):
                    candidates.append(elem)
        if not candidates:
            return None, ""

        elem = random.choice(candidates)
        attr = random.choice(
            [a for a in ("fill", "stroke") if elem.get(a) and elem.get(a) != "none"]
            or ["fill"]
        )
        old_val = elem.get(attr, "none")
        new_val = _random_hex_color()
        # Ensure actually different
        while new_val.lower() == old_val.lower():
            new_val = _random_hex_color()

        tag_local = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        desc = f"Changed {attr} of <{tag_local}> from {old_val} to {new_val}"

        new_tree = copy.deepcopy(tree)
        # Find the same element in the copy by matching path
        _apply_to_matching(new_tree, elem, lambda e: e.set(attr, new_val))
        return new_tree, desc

    # ── position ───────────────────────────────────────────────────

    @staticmethod
    def perturb_position(tree: ET.ElementTree) -> Tuple[Optional[ET.ElementTree], str]:
        """Shift x/y coordinates of a visible element by 5-20px."""
        root = tree.getroot()
        coord_attrs = {"x", "y", "cx", "cy", "x1", "y1", "x2", "y2"}
        candidates: List[Tuple[ET.Element, str]] = []
        for elem in root.iter():
            if elem.tag in VISIBLE_TAGS:
                for attr in coord_attrs:
                    val = elem.get(attr)
                    if val is not None:
                        try:
                            float(val)
                            candidates.append((elem, attr))
                        except ValueError:
                            pass
        if not candidates:
            return None, ""

        elem, attr = random.choice(candidates)
        old_val = elem.get(attr)
        new_val = _shift_value(old_val)
        tag_local = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        desc = f"Shifted {attr} of <{tag_local}> from {old_val} to {new_val}"

        new_tree = copy.deepcopy(tree)
        _apply_to_matching(new_tree, elem, lambda e, a=attr, v=new_val: e.set(a, v))
        return new_tree, desc

    # ── size ───────────────────────────────────────────────────────

    @staticmethod
    def perturb_size(tree: ET.ElementTree) -> Tuple[Optional[ET.ElementTree], str]:
        """Scale width/height or radius of a visible element by 0.7-1.3x."""
        root = tree.getroot()
        size_attrs = {"width", "height", "r", "rx", "ry"}
        candidates: List[Tuple[ET.Element, str]] = []
        for elem in root.iter():
            if elem.tag in VISIBLE_TAGS:
                for attr in size_attrs:
                    val = elem.get(attr)
                    if val is not None:
                        try:
                            float(val)
                            candidates.append((elem, attr))
                        except ValueError:
                            pass
        if not candidates:
            return None, ""

        elem, attr = random.choice(candidates)
        old_val = elem.get(attr)
        scale = random.uniform(0.7, 1.3)
        # Avoid near-identity scales
        while 0.95 < scale < 1.05:
            scale = random.uniform(0.7, 1.3)
        new_val = str(round(float(old_val) * scale, 2))
        tag_local = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        desc = f"Scaled {attr} of <{tag_local}> from {old_val} to {new_val} ({scale:.2f}x)"

        new_tree = copy.deepcopy(tree)
        _apply_to_matching(new_tree, elem, lambda e, a=attr, v=new_val: e.set(a, v))
        return new_tree, desc

    # ── element deletion ──────────────────────────────────────────

    @staticmethod
    def perturb_element(tree: ET.ElementTree) -> Tuple[Optional[ET.ElementTree], str]:
        """Delete a random visible element."""
        root = tree.getroot()
        # Build parent map
        parent_map: Dict[ET.Element, ET.Element] = {}
        for parent in root.iter():
            for child in parent:
                parent_map[child] = parent

        candidates = [e for e in root.iter() if e.tag in VISIBLE_TAGS and e in parent_map]
        if not candidates:
            return None, ""

        elem = random.choice(candidates)
        tag_local = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        elem_id = elem.get("id", "")
        id_info = f" id='{elem_id}'" if elem_id else ""
        desc = f"Deleted <{tag_local}>{id_info} element"

        new_tree = copy.deepcopy(tree)
        new_root = new_tree.getroot()
        new_parent_map: Dict[ET.Element, ET.Element] = {}
        for p in new_root.iter():
            for c in p:
                new_parent_map[c] = p
        # Find matching element by index in iteration order
        orig_idx = list(root.iter()).index(elem)
        new_elem = list(new_root.iter())[orig_idx]
        if new_elem in new_parent_map:
            new_parent_map[new_elem].remove(new_elem)

        return new_tree, desc

    # ── text ──────────────────────────────────────────────────────

    @staticmethod
    def perturb_text(tree: ET.ElementTree) -> Tuple[Optional[ET.ElementTree], str]:
        """Modify text content of a <text> or <tspan> element."""
        TEXT_TAG = f"{{{SVG_NS}}}text"
        TSPAN_TAG = f"{{{SVG_NS}}}tspan"
        root = tree.getroot()

        candidates: List[ET.Element] = []
        for elem in root.iter():
            if elem.tag in (TEXT_TAG, TSPAN_TAG) and elem.text and len(elem.text.strip()) > 0:
                candidates.append(elem)
        if not candidates:
            return None, ""

        elem = random.choice(candidates)
        old_text = elem.text
        tag_local = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

        strategy = random.choice(["char_replace", "truncate", "swap"])

        if strategy == "char_replace" and len(old_text) >= 1:
            text_list = list(old_text)
            n_replace = min(random.randint(1, 3), len(text_list))
            indices = random.sample(range(len(text_list)), n_replace)
            for idx in indices:
                pool = string.ascii_letters + string.digits
                replacement = random.choice(pool)
                while replacement == text_list[idx]:
                    replacement = random.choice(pool)
                text_list[idx] = replacement
            new_text = "".join(text_list)
        elif strategy == "truncate" and len(old_text) >= 2:
            cut_ratio = random.uniform(0.2, 0.5)
            cut_pos = max(1, int(len(old_text) * (1 - cut_ratio)))
            new_text = old_text[:cut_pos]
        elif strategy == "swap" and len(old_text) >= 2:
            text_list = list(old_text)
            i, j = random.sample(range(len(text_list)), 2)
            text_list[i], text_list[j] = text_list[j], text_list[i]
            new_text = "".join(text_list)
        else:
            # Fallback: single char replace
            text_list = list(old_text)
            idx = random.randint(0, len(text_list) - 1)
            text_list[idx] = random.choice(string.ascii_letters)
            new_text = "".join(text_list)

        if new_text == old_text:
            new_text = old_text + "?"

        desc = f"Changed text content of <{tag_local}> from '{old_text}' to '{new_text}'"

        new_tree = copy.deepcopy(tree)
        _apply_to_matching(new_tree, elem, lambda e: setattr(e, "text", new_text))
        return new_tree, desc

    # ── style ─────────────────────────────────────────────────────

    @staticmethod
    def perturb_style(tree: ET.ElementTree) -> Tuple[Optional[ET.ElementTree], str]:
        """Modify visual style attributes (non-color) of an element."""
        TEXT_TAG = f"{{{SVG_NS}}}text"
        root = tree.getroot()

        strategies = []

        # Collect candidates per strategy
        text_elems = [e for e in root.iter() if e.tag == TEXT_TAG]
        visible_elems = [e for e in root.iter() if e.tag in VISIBLE_TAGS]
        stroked_elems = [e for e in visible_elems if e.get("stroke") and e.get("stroke") != "none"]

        if text_elems:
            strategies.append("font_family")
        if stroked_elems:
            strategies.append("stroke_dasharray")
            strategies.append("stroke_width")
        if visible_elems:
            strategies.append("opacity")
        filled_elems = [e for e in visible_elems if e.get("fill") and e.get("fill") != "none"]
        if filled_elems:
            strategies.append("fill_rule")

        if not strategies:
            return None, ""

        strategy = random.choice(strategies)

        if strategy == "font_family":
            elem = random.choice(text_elems)
            old_val = elem.get("font-family", "serif")
            families = ["serif", "sans-serif", "monospace", "cursive", "fantasy"]
            families = [f for f in families if f != old_val]
            new_val = random.choice(families)
            tag_local = "text"
            desc = f"Changed font-family of <{tag_local}> from '{old_val}' to '{new_val}'"
            attr_name = "font-family"

            new_tree = copy.deepcopy(tree)
            _apply_to_matching(new_tree, elem, lambda e: e.set(attr_name, new_val))
            return new_tree, desc

        elif strategy == "stroke_dasharray":
            elem = random.choice(stroked_elems)
            old_val = elem.get("stroke-dasharray", "none")
            dash_patterns = ["4 2", "8 4", "2 2", "6 3 2 3", "10 5"]
            dash_patterns = [p for p in dash_patterns if p != old_val]
            new_val = random.choice(dash_patterns)
            tag_local = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            desc = f"Changed stroke-dasharray of <{tag_local}> from '{old_val}' to '{new_val}'"

            new_tree = copy.deepcopy(tree)
            _apply_to_matching(new_tree, elem, lambda e: e.set("stroke-dasharray", new_val))
            return new_tree, desc

        elif strategy == "opacity":
            elem = random.choice(visible_elems)
            old_val = elem.get("opacity", "1.0")
            new_val = str(round(random.uniform(0.3, 0.9), 1))
            while new_val == old_val:
                new_val = str(round(random.uniform(0.3, 0.9), 1))
            tag_local = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            desc = f"Changed opacity of <{tag_local}> from {old_val} to {new_val}"

            new_tree = copy.deepcopy(tree)
            _apply_to_matching(new_tree, elem, lambda e: e.set("opacity", new_val))
            return new_tree, desc

        elif strategy == "fill_rule":
            elem = random.choice(filled_elems)
            old_val = elem.get("fill-rule", "nonzero")
            new_val = "evenodd" if old_val == "nonzero" else "nonzero"
            tag_local = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            desc = f"Changed fill-rule of <{tag_local}> from '{old_val}' to '{new_val}'"

            new_tree = copy.deepcopy(tree)
            _apply_to_matching(new_tree, elem, lambda e: e.set("fill-rule", new_val))
            return new_tree, desc

        elif strategy == "stroke_width":
            elem = random.choice(stroked_elems)
            old_val = elem.get("stroke-width", "1")
            try:
                old_num = float(old_val)
            except ValueError:
                old_num = 1.0
            offset = random.choice([-1, 1]) * random.randint(1, 3)
            new_num = max(0.5, old_num + offset)
            new_val = str(round(new_num, 1))
            tag_local = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            desc = f"Changed stroke-width of <{tag_local}> from {old_val} to {new_val}"

            new_tree = copy.deepcopy(tree)
            _apply_to_matching(new_tree, elem, lambda e: e.set("stroke-width", new_val))
            return new_tree, desc

        return None, ""


def _apply_to_matching(new_tree: ET.ElementTree, orig_elem: ET.Element, fn):
    """Apply fn to the element in new_tree matching orig_elem's position."""
    orig_root_iter = list(new_tree.getroot().iter())
    # We use deepcopy so element order is preserved; find by index from the
    # original tree that was deepcopied.  Since we deepcopy `tree` (which
    # contains orig_elem), find index among the *original* tree's iteration.
    # But we already have the deepcopy — the caller passes the original tree's
    # element. We need to find the same position.
    # Strategy: iterate both trees in lockstep. Since the new tree is a
    # deepcopy made *before* fn, the structures are identical.
    # The caller already deepcopied, so we just search by tag+attrib match.
    for new_elem in new_tree.getroot().iter():
        if (new_elem.tag == orig_elem.tag and
                new_elem.attrib == orig_elem.attrib and
                new_elem.text == orig_elem.text):
            fn(new_elem)
            return


# ── Perturbation dispatch ─────────────────────────────────────────────

PERTURBATION_MAP = {
    ErrorCategory.COLOR: SVGPerturbation.perturb_color,
    ErrorCategory.POSITION: SVGPerturbation.perturb_position,
    ErrorCategory.SIZE: SVGPerturbation.perturb_size,
    ErrorCategory.ELEMENT: SVGPerturbation.perturb_element,
    ErrorCategory.TEXT: SVGPerturbation.perturb_text,
    ErrorCategory.STYLE: SVGPerturbation.perturb_style,
}

PERTURBATION_TYPES = list(PERTURBATION_MAP.keys())


# ── Pipeline ──────────────────────────────────────────────────────────

class SVGPipeline:
    """End-to-end pipeline: load SVGs → perturb → render → output."""

    def __init__(
        self,
        render_width: int = 256,
        render_height: int = 256,
        seed: Optional[int] = None,
    ):
        self.render_width = render_width
        self.render_height = render_height
        if seed is not None:
            random.seed(seed)

    def load_mmsvg_icons(self, max_samples: Optional[int] = None) -> List[str]:
        """Load SVG strings from HuggingFace MMSVG-2M Icon subset."""
        try:
            from datasets import load_dataset
        except ImportError:
            print("[WARN] `datasets` not installed. Use synthetic SVGs instead.")
            return []

        try:
            ds = load_dataset("linxy/MMSVG-2M", split="train", streaming=True)
            svgs: List[str] = []
            for i, sample in enumerate(ds):
                if max_samples and i >= max_samples:
                    break
                # Try common column names
                svg_str = sample.get("svg") or sample.get("code") or sample.get("text")
                if svg_str and "<svg" in svg_str:
                    svgs.append(svg_str)
            if svgs:
                print(f"[INFO] Loaded {len(svgs)} SVGs from MMSVG-2M")
            return svgs
        except Exception as e:
            print(f"[WARN] Failed to load MMSVG-2M: {e}")
            return []

    @staticmethod
    def synthetic_svgs() -> List[str]:
        """Generate a small set of synthetic SVGs for testing."""
        samples = [
            # Simple rect + circle
            '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200">'
            '<rect x="10" y="10" width="80" height="60" fill="#ff0000" stroke="#000000" stroke-width="2"/>'
            '<circle cx="150" cy="100" r="40" fill="#00ff00" stroke="#333333"/>'
            '<text x="50" y="180" fill="#0000ff" font-size="14">Hello SVG</text>'
            '</svg>',
            # Multiple rects
            '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256">'
            '<rect x="20" y="20" width="100" height="100" fill="#3498db"/>'
            '<rect x="140" y="20" width="96" height="96" fill="#e74c3c" rx="10"/>'
            '<rect x="80" y="140" width="96" height="96" fill="#2ecc71"/>'
            '</svg>',
            # Path + ellipse
            '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200">'
            '<ellipse cx="100" cy="80" rx="60" ry="40" fill="#f39c12" stroke="#000" stroke-width="1"/>'
            '<path d="M 20 180 L 100 120 L 180 180 Z" fill="#9b59b6"/>'
            '<rect x="70" y="150" width="60" height="30" fill="#1abc9c"/>'
            '</svg>',
            # Icon-like: simple star
            '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200">'
            '<polygon points="100,10 40,198 190,78 10,78 160,198" fill="#f1c40f" stroke="#e67e22" stroke-width="3"/>'
            '</svg>',
            # Grid of circles
            '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200">'
            '<circle cx="50" cy="50" r="30" fill="#e74c3c"/>'
            '<circle cx="150" cy="50" r="30" fill="#3498db"/>'
            '<circle cx="50" cy="150" r="30" fill="#2ecc71"/>'
            '<circle cx="150" cy="150" r="30" fill="#9b59b6"/>'
            '</svg>',
            # Text-heavy: labeled bar chart
            '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256">'
            '<rect x="30" y="100" width="40" height="120" fill="#3498db" stroke="#2c3e50" stroke-width="1"/>'
            '<rect x="90" y="60" width="40" height="160" fill="#e74c3c" stroke="#2c3e50" stroke-width="1"/>'
            '<rect x="150" y="130" width="40" height="90" fill="#2ecc71" stroke="#2c3e50" stroke-width="1"/>'
            '<text x="50" y="240" font-size="12" fill="#333" text-anchor="middle" font-family="serif">Alpha</text>'
            '<text x="110" y="240" font-size="12" fill="#333" text-anchor="middle" font-family="serif">Beta</text>'
            '<text x="170" y="240" font-size="12" fill="#333" text-anchor="middle" font-family="serif">Gamma</text>'
            '<text x="128" y="30" font-size="16" fill="#000" text-anchor="middle" font-family="sans-serif">Results</text>'
            '</svg>',
            # Text + styled shapes
            '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" viewBox="0 0 200 200">'
            '<rect x="10" y="10" width="180" height="180" fill="none" stroke="#333" stroke-width="2" stroke-dasharray="5 3"/>'
            '<circle cx="100" cy="80" r="35" fill="#e67e22" stroke="#d35400" stroke-width="2"/>'
            '<text x="100" y="85" font-size="20" fill="#fff" text-anchor="middle" font-family="monospace">42</text>'
            '<text x="100" y="150" font-size="14" fill="#555" text-anchor="middle">Answer</text>'
            '</svg>',
        ]
        return samples

    def process_one(
        self, svg_code: str, error_category: Optional[ErrorCategory] = None
    ) -> Optional[Dict[str, Any]]:
        """Apply a random perturbation to one SVG and render both versions.

        Args:
            svg_code: Original SVG string.
            error_category: Force a specific perturbation type, or random if None.

        Returns:
            Dict with original/perturbed SVG, images, error_type, description.
            None if perturbation fails (e.g. no suitable elements).
        """
        try:
            tree = ET.ElementTree(ET.fromstring(svg_code))
        except ET.ParseError:
            return None

        if error_category is None:
            error_category = random.choice(PERTURBATION_TYPES)

        perturb_fn = PERTURBATION_MAP[error_category]
        result = perturb_fn(tree)
        if result is None:
            return None
        new_tree, description = result
        if new_tree is None:
            return None

        # Serialize back to strings
        perturbed_svg = ET.tostring(new_tree.getroot(), encoding="unicode")
        # Ensure XML declaration / SVG wrapper is present
        if not perturbed_svg.strip().startswith("<?xml"):
            perturbed_svg = '<?xml version="1.0" encoding="UTF-8"?>\n' + perturbed_svg

        original_svg = svg_code
        if not original_svg.strip().startswith("<?xml"):
            original_svg = '<?xml version="1.0" encoding="UTF-8"?>\n' + original_svg

        try:
            original_img = render_svg(original_svg, self.render_width, self.render_height)
            perturbed_img = render_svg(perturbed_svg, self.render_width, self.render_height)
        except Exception:
            return None

        return {
            "original_svg": original_svg,
            "perturbed_svg": perturbed_svg,
            "original_img": original_img,
            "perturbed_img": perturbed_img,
            "error_type": error_category.value,
            "error_description": description,
        }

    def run(
        self,
        num_samples: int = 100,
        output_dir: Optional[str] = None,
        use_hf: bool = True,
    ) -> List[Dict[str, Any]]:
        """Run the full pipeline.

        Args:
            num_samples: Number of perturbed pairs to generate.
            output_dir: If set, save images and metadata to this directory.
            use_hf: Whether to attempt loading from HuggingFace first.

        Returns:
            List of sample dicts.
        """
        # Load SVG sources
        svgs: List[str] = []
        if use_hf:
            svgs = self.load_mmsvg_icons(max_samples=num_samples * 2)
        if not svgs:
            print("[INFO] Using synthetic SVGs for pipeline.")
            svgs = self.synthetic_svgs()

        results: List[Dict[str, Any]] = []
        attempts = 0
        max_attempts = num_samples * 5

        while len(results) < num_samples and attempts < max_attempts:
            svg = random.choice(svgs)
            sample = self.process_one(svg)
            if sample is not None:
                results.append(sample)
            attempts += 1

        print(f"[INFO] Generated {len(results)}/{num_samples} samples ({attempts} attempts)")

        if output_dir:
            self._save(results, output_dir)

        return results

    def _save(self, results: List[Dict[str, Any]], output_dir: str):
        """Save results to disk."""
        out = Path(output_dir)
        (out / "original_imgs").mkdir(parents=True, exist_ok=True)
        (out / "perturbed_imgs").mkdir(parents=True, exist_ok=True)

        metadata = []
        for i, sample in enumerate(results):
            orig_path = out / "original_imgs" / f"{i:05d}.png"
            pert_path = out / "perturbed_imgs" / f"{i:05d}.png"
            sample["original_img"].save(str(orig_path))
            sample["perturbed_img"].save(str(pert_path))

            metadata.append({
                "id": i,
                "error_type": sample["error_type"],
                "error_description": sample["error_description"],
                "original_svg": sample["original_svg"],
                "perturbed_svg": sample["perturbed_svg"],
                "original_img_path": str(orig_path),
                "perturbed_img_path": str(pert_path),
            })

        meta_path = out / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Saved {len(results)} samples to {output_dir}")


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SVG perturbation pipeline for training data generation."
    )
    parser.add_argument(
        "--num_samples", type=int, default=100,
        help="Number of perturbed pairs to generate (default: 100)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/processed/svg/",
        help="Output directory for images and metadata (default: data/processed/svg/)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--width", type=int, default=256,
        help="Render width in pixels (default: 256)",
    )
    parser.add_argument(
        "--height", type=int, default=256,
        help="Render height in pixels (default: 256)",
    )
    parser.add_argument(
        "--no_hf", action="store_true",
        help="Skip HuggingFace dataset, use synthetic SVGs only",
    )
    args = parser.parse_args()

    pipeline = SVGPipeline(
        render_width=args.width,
        render_height=args.height,
        seed=args.seed,
    )
    pipeline.run(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        use_hf=not args.no_hf,
    )


if __name__ == "__main__":
    main()
