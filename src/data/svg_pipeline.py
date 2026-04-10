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
    def synthetic_svgs(num_templates: int = 500) -> List[str]:
        """Generate diverse synthetic SVGs procedurally.

        Produces structurally unique SVGs by combining different element types,
        counts, and layout patterns. Each call generates `num_templates` SVGs.
        """
        import math

        COLORS = [
            "#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f1c40f",
            "#e67e22", "#1abc9c", "#34495e", "#c0392b", "#2980b9",
            "#27ae60", "#8e44ad", "#f39c12", "#d35400", "#16a085",
            "#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#ffeaa7",
            "#dfe6e9", "#636e72", "#fd79a8", "#6c5ce7", "#00b894",
        ]
        STROKE_COLORS = ["#000000", "#333333", "#2c3e50", "#555555", "none"]
        FONTS = ["serif", "sans-serif", "monospace", "cursive"]
        WORDS = [
            "Alpha", "Beta", "Gamma", "Delta", "Hello", "World", "Test",
            "Data", "Node", "Edge", "Peak", "Flow", "Core", "Link", "Wave",
            "Grid", "Axis", "Plot", "Item", "Cell", "Loop", "Path", "Area",
            "100", "42", "7.5", "OK", "A", "B", "C", "X", "Y", "Z",
        ]

        # Element generators (each returns an SVG element string)
        def _rect(x, y, w, h, fill, stroke="none", sw=1, rx=0):
            rx_attr = f' rx="{rx}"' if rx else ""
            sw_attr = f' stroke="{stroke}" stroke-width="{sw}"' if stroke != "none" else ""
            return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}"{sw_attr}{rx_attr}/>'

        def _circle(cx, cy, r, fill, stroke="none", sw=1):
            sw_attr = f' stroke="{stroke}" stroke-width="{sw}"' if stroke != "none" else ""
            return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"{sw_attr}/>'

        def _ellipse(cx, cy, rx, ry, fill, stroke="none", sw=1):
            sw_attr = f' stroke="{stroke}" stroke-width="{sw}"' if stroke != "none" else ""
            return f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{fill}"{sw_attr}/>'

        def _line(x1, y1, x2, y2, stroke, sw=2):
            return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"/>'

        def _polygon(points_str, fill, stroke="none", sw=1):
            sw_attr = f' stroke="{stroke}" stroke-width="{sw}"' if stroke != "none" else ""
            return f'<polygon points="{points_str}" fill="{fill}"{sw_attr}/>'

        def _path(d, fill, stroke="none", sw=1):
            fill_attr = f'fill="{fill}"'
            sw_attr = f' stroke="{stroke}" stroke-width="{sw}"' if stroke != "none" else ""
            return f'<path d="{d}" {fill_attr}{sw_attr}/>'

        def _text(x, y, content, fill="#333", font_size=14, font_family="serif", anchor="middle"):
            return (f'<text x="{x}" y="{y}" font-size="{font_size}" fill="{fill}" '
                    f'text-anchor="{anchor}" font-family="{font_family}">{content}</text>')

        def _random_triangle(cx, cy, size):
            pts = []
            for a in [0, 120, 240]:
                rad = math.radians(a - 90)
                pts.append(f"{cx + size * math.cos(rad):.0f},{cy + size * math.sin(rad):.0f}")
            return " ".join(pts)

        def _random_polygon_pts(cx, cy, r, n_sides):
            pts = []
            for i in range(n_sides):
                angle = 2 * math.pi * i / n_sides - math.pi / 2
                pts.append(f"{cx + r * math.cos(angle):.0f},{cy + r * math.sin(angle):.0f}")
            return " ".join(pts)

        # Layout generators: each returns a list of element strings
        def gen_scattered_shapes(rng):
            """2-8 random shapes scattered across the canvas."""
            n = rng.randint(2, 8)
            elems = []
            shape_types = ["rect", "circle", "ellipse", "polygon", "line"]
            for _ in range(n):
                st = rng.choice(shape_types)
                c = rng.choice(COLORS)
                sc = rng.choice(STROKE_COLORS)
                if st == "rect":
                    x, y = rng.randint(10, 180), rng.randint(10, 180)
                    w, h = rng.randint(20, 80), rng.randint(20, 80)
                    rx = rng.choice([0, 0, 5, 10])
                    elems.append(_rect(x, y, w, h, c, sc, rng.randint(1, 3), rx))
                elif st == "circle":
                    cx, cy = rng.randint(30, 220), rng.randint(30, 220)
                    r = rng.randint(10, 50)
                    elems.append(_circle(cx, cy, r, c, sc))
                elif st == "ellipse":
                    cx, cy = rng.randint(30, 220), rng.randint(30, 220)
                    rx, ry = rng.randint(15, 60), rng.randint(10, 40)
                    elems.append(_ellipse(cx, cy, rx, ry, c, sc))
                elif st == "polygon":
                    cx, cy = rng.randint(40, 210), rng.randint(40, 210)
                    r = rng.randint(15, 50)
                    sides = rng.choice([3, 5, 6])
                    pts = _random_polygon_pts(cx, cy, r, sides)
                    elems.append(_polygon(pts, c, sc, rng.randint(1, 2)))
                else:  # line
                    elems.append(_line(rng.randint(5, 240), rng.randint(5, 240),
                                       rng.randint(5, 240), rng.randint(5, 240),
                                       sc if sc != "none" else c, rng.randint(1, 3)))
            return elems

        def gen_grid(rng):
            """2x2, 2x3, 3x2, or 3x3 grid with mixed element types."""
            cols = rng.choice([2, 3])
            rows = rng.choice([2, 3])
            cell_types = ["rect", "circle", "ellipse", "polygon"]
            # Either uniform or mixed types per cell
            mixed = rng.random() > 0.3
            base_type = rng.choice(cell_types)
            gap = 256 // (max(cols, rows) + 1)
            elems = []
            for r in range(rows):
                for c_idx in range(cols):
                    color = rng.choice(COLORS)
                    sc = rng.choice(STROKE_COLORS)
                    x = gap * (c_idx + 1) - gap // 3
                    y = gap * (r + 1) - gap // 3
                    st = rng.choice(cell_types) if mixed else base_type
                    if st == "rect":
                        elems.append(_rect(x, y, gap // 2, gap // 2, color, sc))
                    elif st == "circle":
                        elems.append(_circle(x + gap // 4, y + gap // 4, gap // 4, color, sc))
                    elif st == "ellipse":
                        elems.append(_ellipse(x + gap // 4, y + gap // 4,
                                              gap // 3, gap // 5, color, sc))
                    else:
                        pts = _random_polygon_pts(x + gap // 4, y + gap // 4,
                                                   gap // 4, rng.choice([3, 5, 6]))
                        elems.append(_polygon(pts, color, sc))
            # Optional title
            if rng.random() > 0.5:
                elems.append(_text(128, 20, rng.choice(WORDS), rng.choice(COLORS),
                                   rng.choice([12, 14, 16]), rng.choice(FONTS)))
            return elems

        def gen_bar_chart(rng):
            """Bar chart with 2-6 bars and optional labels."""
            n_bars = rng.randint(2, 6)
            bar_w = max(15, 200 // n_bars - 10)
            elems = []
            for i in range(n_bars):
                h = rng.randint(30, 180)
                x = 20 + i * (bar_w + 8)
                y = 230 - h
                color = rng.choice(COLORS)
                sc = rng.choice(STROKE_COLORS[:3])
                elems.append(_rect(x, y, bar_w, h, color, sc, 1))
                if rng.random() > 0.3:
                    label = rng.choice(WORDS)
                    elems.append(_text(x + bar_w // 2, 248, label, font_size=rng.choice([10, 12])))
            if rng.random() > 0.5:
                elems.append(_text(128, 20, rng.choice(WORDS), font_size=16, font_family="sans-serif"))
            return elems

        def gen_concentric(rng):
            """2-5 concentric circles or ellipses, optionally with label/line."""
            n = rng.randint(2, 5)
            cx, cy = rng.randint(80, 170), rng.randint(80, 170)
            max_r = rng.randint(50, 90)
            elems = []
            for i in range(n):
                r = max_r - i * (max_r // n)
                if r < 5:
                    break
                color = rng.choice(COLORS)
                shape = rng.choice(["circle", "ellipse"])
                if shape == "circle":
                    elems.append(_circle(cx, cy, r, color, rng.choice(STROKE_COLORS)))
                else:
                    elems.append(_ellipse(cx, cy, r, int(r * rng.uniform(0.5, 0.9)), color))
            # Optionally add a label or crosshair
            if rng.random() > 0.5:
                elems.append(_text(cx, cy, rng.choice(WORDS), rng.choice(COLORS),
                                   rng.choice([10, 12, 14]), rng.choice(FONTS)))
            if rng.random() > 0.6:
                elems.append(_line(cx - max_r, cy, cx + max_r, cy,
                                   rng.choice(STROKE_COLORS[:3]), 1))
            return elems

        def gen_polygon_scene(rng):
            """1-4 polygons with optional labels and decorative elements."""
            n = rng.randint(1, 4)
            elems = []
            for _ in range(n):
                cx, cy = rng.randint(40, 210), rng.randint(40, 210)
                r = rng.randint(20, 60)
                sides = rng.choice([3, 4, 5, 6, 8])
                pts = _random_polygon_pts(cx, cy, r, sides)
                color = rng.choice(COLORS)
                sc = rng.choice(STROKE_COLORS)
                elems.append(_polygon(pts, color, sc, rng.randint(1, 3)))
                if rng.random() > 0.6:
                    elems.append(_text(cx, cy, rng.choice(WORDS), rng.choice(COLORS),
                                       rng.choice([10, 12]), rng.choice(FONTS)))
            if rng.random() > 0.5:
                elems.append(_line(rng.randint(5, 50), rng.randint(5, 240),
                                   rng.randint(200, 250), rng.randint(5, 240),
                                   rng.choice(STROKE_COLORS[:3]), rng.randint(1, 2)))
            return elems

        def gen_path_scene(rng):
            """1-4 paths with optional shapes and labels."""
            path_templates = [
                lambda: f"M {rng.randint(10,80)} {rng.randint(150,230)} L {rng.randint(80,170)} {rng.randint(30,100)} L {rng.randint(170,240)} {rng.randint(150,230)} Z",
                lambda: f"M {rng.randint(10,50)} {rng.randint(100,200)} Q {rng.randint(100,150)} {rng.randint(10,60)} {rng.randint(190,240)} {rng.randint(100,200)}",
                lambda: f"M {rng.randint(10,30)} {rng.randint(120,180)} L {rng.randint(60,80)} {rng.randint(40,80)} L {rng.randint(120,160)} {rng.randint(120,180)} L {rng.randint(180,220)} {rng.randint(40,80)}",
                lambda: f"M {rng.randint(10,50)} {rng.randint(10,50)} C {rng.randint(60,120)} {rng.randint(60,120)} {rng.randint(130,190)} {rng.randint(60,120)} {rng.randint(200,240)} {rng.randint(10,50)}",
            ]
            n = rng.randint(1, 4)
            elems = []
            for _ in range(n):
                d = rng.choice(path_templates)()
                color = rng.choice(COLORS)
                sc = rng.choice(STROKE_COLORS)
                elems.append(_path(d, color, sc, rng.randint(1, 3)))
            # Optionally mix in a shape or text
            if rng.random() > 0.5:
                c = rng.choice(COLORS)
                elems.append(_circle(rng.randint(30, 220), rng.randint(30, 220),
                                     rng.randint(5, 20), c))
            if rng.random() > 0.6:
                elems.append(_text(rng.randint(20, 200), rng.randint(20, 240),
                                   rng.choice(WORDS), rng.choice(COLORS),
                                   rng.choice([10, 12]), rng.choice(FONTS)))
            return elems

        def gen_lines_and_shapes(rng):
            """Mix of lines and basic shapes."""
            elems = []
            n_lines = rng.randint(1, 4)
            for _ in range(n_lines):
                elems.append(_line(
                    rng.randint(10, 240), rng.randint(10, 240),
                    rng.randint(10, 240), rng.randint(10, 240),
                    rng.choice(COLORS), rng.randint(1, 4),
                ))
            n_shapes = rng.randint(1, 3)
            for _ in range(n_shapes):
                c = rng.choice(COLORS)
                if rng.random() > 0.5:
                    elems.append(_rect(rng.randint(10, 150), rng.randint(10, 150),
                                       rng.randint(20, 80), rng.randint(20, 80), c))
                else:
                    elems.append(_circle(rng.randint(30, 220), rng.randint(30, 220),
                                         rng.randint(10, 40), c))
            return elems

        def gen_labeled_diagram(rng):
            """Shapes with text labels, varied element types."""
            elems = []
            n = rng.randint(1, 5)
            shape_choice = rng.choice(["circle", "rect", "ellipse", "polygon", "mixed"])
            for _ in range(n):
                cx, cy = rng.randint(40, 210), rng.randint(40, 200)
                c = rng.choice(COLORS)
                sc = rng.choice(STROKE_COLORS)
                st = rng.choice(["circle", "rect", "ellipse", "polygon"]) if shape_choice == "mixed" else shape_choice
                if st == "circle":
                    r = rng.randint(15, 40)
                    elems.append(_circle(cx, cy, r, c, sc))
                elif st == "rect":
                    w, h = rng.randint(30, 80), rng.randint(20, 50)
                    elems.append(_rect(cx - w // 2, cy - h // 2, w, h, c, sc, 1,
                                       rng.choice([0, 5])))
                elif st == "ellipse":
                    elems.append(_ellipse(cx, cy, rng.randint(20, 50), rng.randint(12, 30), c, sc))
                else:
                    pts = _random_polygon_pts(cx, cy, rng.randint(15, 35), rng.choice([3, 5, 6]))
                    elems.append(_polygon(pts, c, sc))
                # Label (sometimes skip for variety)
                if rng.random() > 0.2:
                    elems.append(_text(cx, cy + 5, rng.choice(WORDS),
                                       "#fff" if rng.random() > 0.5 else "#333",
                                       rng.choice([10, 12, 14]), rng.choice(FONTS)))
            # Optional connecting lines
            if n >= 2 and rng.random() > 0.5:
                elems.append(_line(rng.randint(30, 120), rng.randint(30, 220),
                                   rng.randint(130, 230), rng.randint(30, 220),
                                   rng.choice(STROKE_COLORS[:3]), 1))
            return elems

        def gen_mixed_complex(rng):
            """3-7 elements mixing multiple types."""
            n = rng.randint(3, 7)
            makers = [
                lambda: _rect(rng.randint(5, 180), rng.randint(5, 180),
                              rng.randint(15, 90), rng.randint(15, 90),
                              rng.choice(COLORS), rng.choice(STROKE_COLORS)),
                lambda: _circle(rng.randint(20, 230), rng.randint(20, 230),
                                rng.randint(8, 50), rng.choice(COLORS)),
                lambda: _ellipse(rng.randint(30, 220), rng.randint(30, 220),
                                 rng.randint(10, 50), rng.randint(8, 35), rng.choice(COLORS)),
                lambda: _polygon(_random_polygon_pts(rng.randint(40, 200), rng.randint(40, 200),
                                                      rng.randint(15, 50), rng.choice([3, 5, 6])),
                                 rng.choice(COLORS), rng.choice(STROKE_COLORS)),
                lambda: _text(rng.randint(20, 230), rng.randint(20, 240), rng.choice(WORDS),
                              rng.choice(COLORS), rng.choice([10, 12, 14, 16]), rng.choice(FONTS)),
                lambda: _line(rng.randint(5, 240), rng.randint(5, 240),
                              rng.randint(5, 240), rng.randint(5, 240),
                              rng.choice(COLORS), rng.randint(1, 4)),
            ]
            elems = []
            for _ in range(n):
                elems.append(rng.choice(makers)())
            return elems

        # Layout registry
        LAYOUTS = [
            gen_scattered_shapes,
            gen_grid,
            gen_bar_chart,
            gen_concentric,
            gen_polygon_scene,
            gen_path_scene,
            gen_lines_and_shapes,
            gen_labeled_diagram,
            gen_mixed_complex,
        ]

        # Generate diverse templates
        samples = []
        rng = random.Random(42)  # Deterministic for reproducibility
        for i in range(num_templates):
            layout_fn = LAYOUTS[i % len(LAYOUTS)]
            # Use different seed per template for variation within same layout
            sub_rng = random.Random(42 + i * 137)
            elems = layout_fn(sub_rng)
            size = sub_rng.choice([200, 224, 256])
            svg = (f'<svg xmlns="http://www.w3.org/2000/svg" '
                   f'width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
                   + "".join(elems) + "</svg>")
            samples.append(svg)

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
        svg_idx = 0

        while len(results) < num_samples and attempts < max_attempts:
            svg = svgs[svg_idx % len(svgs)]
            svg_idx += 1
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
        if not results:
            print("[WARN] No results to save, skipping metadata write")
            return
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
