"""HTML perturbation pipeline for generating training pairs.

Generates (original_html, perturbed_html, original_img, perturbed_img,
error_type, error_description) tuples for cross-format visual refinement.

6 error categories aligned with error_taxonomy.py:
  COLOR, POSITION, SIZE, ELEMENT, TEXT, STYLE
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
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup, Tag
from PIL import Image

# Allow running as script from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.error_taxonomy import ErrorCategory


# ── Helpers ──────────────────────────────────────────────────────────

def _random_hex_color() -> str:
    return f"#{random.randint(0, 0xFFFFFF):06x}"


def _parse_style(style_str: str) -> Dict[str, str]:
    props = {}
    if not style_str:
        return props
    for part in style_str.split(";"):
        part = part.strip()
        if ":" in part:
            key, val = part.split(":", 1)
            props[key.strip()] = val.strip()
    return props


def _serialize_style(props: Dict[str, str]) -> str:
    return "; ".join(f"{k}: {v}" for k, v in props.items())


def _set_style_prop(tag: Tag, prop: str, value: str):
    props = _parse_style(tag.get("style", ""))
    props[prop] = value
    tag["style"] = _serialize_style(props)


SKIP_TAGS = {"script", "style", "meta", "link", "head", "title", "html",
             "[document]", "noscript", "br", "hr"}


def _get_visible_elements(soup: BeautifulSoup) -> List[Tag]:
    return [t for t in soup.find_all(True) if t.name not in SKIP_TAGS]


def _find_styled(soup, css_props):
    """Find elements with any of given CSS props in inline style."""
    results = []
    for tag in _get_visible_elements(soup):
        style = _parse_style(tag.get("style", ""))
        for prop in css_props:
            if prop in style:
                results.append((tag, prop, style[prop]))
    return results


def _find_matching_tag(new_soup: BeautifulSoup, orig_tag: Tag) -> Optional[Tag]:
    """Find corresponding element in copied soup by tree position."""
    path = []
    current = orig_tag
    while current.parent is not None and current.parent.name != "[document]":
        parent = current.parent
        children = [c for c in parent.children if isinstance(c, Tag)]
        idx = next((i for i, c in enumerate(children) if c is current), 0)
        path.append(idx)
        current = parent
    path.reverse()

    node = new_soup
    for idx in path:
        children = [c for c in node.children if isinstance(c, Tag)]
        if idx >= len(children):
            return None
        node = children[idx]
    return node if isinstance(node, Tag) else None


# ── Rendering ────────────────────────────────────────────────────────

def render_html_sync(html_code: str, width: int = 1280, height: int = 720) -> Image.Image:
    """Render HTML to PIL Image via wkhtmltoimage (fallback to html2image)."""
    import subprocess

    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = os.path.join(tmpdir, "input.html")
        img_path = os.path.join(tmpdir, "shot.png")
        with open(html_path, "w") as f:
            f.write(html_code)
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
            check=False,
            timeout=30,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not os.path.exists(img_path):
            raise RuntimeError("wkhtmltoimage produced no output")
        return Image.open(img_path).convert("RGB").copy()


# ── Perturbation engine ─────────────────────────────────────────────

class HTMLPerturbation:
    """6 perturbation types aligned with ErrorCategory."""

    # ── COLOR ─────────────────────────────────────────────────────

    @staticmethod
    def perturb_color(soup: BeautifulSoup) -> Tuple[Optional[BeautifulSoup], str]:
        color_props = ["background-color", "color", "border-color"]
        candidates = _find_styled(soup, color_props)
        if not candidates:
            # Add color to a random element
            visible = _get_visible_elements(soup)
            if not visible:
                return None, ""
            tag = random.choice(visible)
            new_soup = copy.deepcopy(soup)
            t = _find_matching_tag(new_soup, tag)
            if not t:
                return None, ""
            new_val = _random_hex_color()
            _set_style_prop(t, "color", new_val)
            return new_soup, f"Added color: {new_val} to <{tag.name}>"

        tag, prop, old_val = random.choice(candidates)
        new_val = _random_hex_color()
        while new_val.lower() == old_val.lower():
            new_val = _random_hex_color()
        new_soup = copy.deepcopy(soup)
        t = _find_matching_tag(new_soup, tag)
        if not t:
            return None, ""
        _set_style_prop(t, prop, new_val)
        return new_soup, f"Changed {prop} of <{tag.name}> from '{old_val}' to '{new_val}'"

    # ── POSITION ─────────────────────────────────────────────────

    @staticmethod
    def perturb_position(soup: BeautifulSoup) -> Tuple[Optional[BeautifulSoup], str]:
        pos_props = ["margin", "margin-top", "margin-bottom", "margin-left",
                     "margin-right", "padding", "padding-top", "padding-bottom",
                     "padding-left", "padding-right", "top", "left", "gap",
                     "flex-direction", "text-align", "justify-content", "align-items"]
        candidates = _find_styled(soup, pos_props)

        if not candidates:
            visible = _get_visible_elements(soup)
            if not visible:
                return None, ""
            tag = random.choice(visible)
            offset = random.randint(10, 40)
            prop = random.choice(["margin-left", "margin-top", "padding-top"])
            new_soup = copy.deepcopy(soup)
            t = _find_matching_tag(new_soup, tag)
            if not t:
                return None, ""
            _set_style_prop(t, prop, f"{offset}px")
            return new_soup, f"Added {prop}: {offset}px to <{tag.name}>"

        tag, prop, old_val = random.choice(candidates)
        new_soup = copy.deepcopy(soup)
        t = _find_matching_tag(new_soup, tag)
        if not t:
            return None, ""

        if prop == "flex-direction":
            opts = [o for o in ["row", "column", "row-reverse", "column-reverse"] if o != old_val]
            new_val = random.choice(opts)
        elif prop in ("text-align",):
            opts = [o for o in ["left", "center", "right", "justify"] if o != old_val]
            new_val = random.choice(opts)
        elif prop in ("justify-content",):
            opts = [o for o in ["flex-start", "flex-end", "center", "space-between", "space-around"] if o != old_val]
            new_val = random.choice(opts)
        elif prop in ("align-items",):
            opts = [o for o in ["flex-start", "flex-end", "center", "stretch", "baseline"] if o != old_val]
            new_val = random.choice(opts)
        else:
            match = re.search(r"-?\d+", old_val)
            if match:
                num = int(match.group())
                offset = random.choice([-1, 1]) * random.randint(10, 30)
                new_val = re.sub(r"-?\d+", str(max(0, num + offset)), old_val, count=1)
            else:
                new_val = f"{random.randint(10, 40)}px"

        _set_style_prop(t, prop, new_val)
        return new_soup, f"Changed {prop} of <{tag.name}> from '{old_val}' to '{new_val}'"

    # ── SIZE ──────────────────────────────────────────────────────

    @staticmethod
    def perturb_size(soup: BeautifulSoup) -> Tuple[Optional[BeautifulSoup], str]:
        size_props = ["width", "height", "font-size", "max-width", "min-height",
                      "border-width", "line-height"]
        candidates = _find_styled(soup, size_props)

        # Also check width/height HTML attributes
        for tag in _get_visible_elements(soup):
            for attr in ("width", "height"):
                val = tag.get(attr)
                if val and re.match(r"\d+", str(val)):
                    candidates.append((tag, f"attr:{attr}", str(val)))

        if not candidates:
            return None, ""

        tag, prop, old_val = random.choice(candidates)
        match = re.search(r"(\d+(?:\.\d+)?)", old_val)
        if not match:
            return None, ""

        num = float(match.group(1))
        scale = random.uniform(0.5, 1.5)
        while 0.9 < scale < 1.1:
            scale = random.uniform(0.5, 1.5)
        new_num = max(1, num * scale)

        new_soup = copy.deepcopy(soup)
        t = _find_matching_tag(new_soup, tag)
        if not t:
            return None, ""

        if prop.startswith("attr:"):
            attr_name = prop.split(":")[1]
            new_val = str(int(new_num))
            t[attr_name] = new_val
            return new_soup, f"Scaled {attr_name} of <{tag.name}> from {old_val} to {new_val} ({scale:.2f}x)"
        else:
            new_val = re.sub(r"\d+(?:\.\d+)?", str(int(new_num)), old_val, count=1)
            _set_style_prop(t, prop, new_val)
            return new_soup, f"Scaled {prop} of <{tag.name}> from '{old_val}' to '{new_val}' ({scale:.2f}x)"

    # ── ELEMENT ──────────────────────────────────────────────────

    @staticmethod
    def perturb_element(soup: BeautifulSoup) -> Tuple[Optional[BeautifulSoup], str]:
        removable = {"button", "input", "select", "textarea", "img", "a", "span",
                     "div", "p", "li", "h1", "h2", "h3", "h4", "h5", "h6",
                     "table", "form", "nav", "section"}
        candidates = [t for t in _get_visible_elements(soup) if t.name in removable and t.parent]
        if not candidates:
            return None, ""

        tag = random.choice(candidates)
        new_soup = copy.deepcopy(soup)
        t = _find_matching_tag(new_soup, tag)
        if not t:
            return None, ""

        action = random.choice(["delete", "replace"])
        if action == "delete":
            text_preview = (t.get_text(strip=True) or "")[:30]
            name = t.name
            t.decompose()
            desc = f"Deleted <{name}> element"
            if text_preview:
                desc += f" containing '{text_preview}'"
        else:
            old_name = t.name
            replacements = {"button": "span", "div": "section", "p": "div",
                            "h1": "p", "h2": "p", "span": "div", "a": "span",
                            "input": "div", "img": "div", "li": "span"}
            new_name = replacements.get(old_name, "div")
            t.name = new_name
            desc = f"Replaced <{old_name}> with <{new_name}>"

        return new_soup, desc

    # ── TEXT ──────────────────────────────────────────────────────

    @staticmethod
    def perturb_text(soup: BeautifulSoup) -> Tuple[Optional[BeautifulSoup], str]:
        candidates = [t for t in _get_visible_elements(soup)
                      if t.string and len(t.string.strip()) >= 2]
        if not candidates:
            return None, ""

        tag = random.choice(candidates)
        old_text = tag.string.strip()

        new_soup = copy.deepcopy(soup)
        t = _find_matching_tag(new_soup, tag)
        if not t:
            return None, ""

        strategy = random.choice(["typo", "truncate", "swap"])
        if strategy == "typo":
            text_list = list(old_text)
            n = min(random.randint(1, 3), len(text_list))
            for idx in random.sample(range(len(text_list)), n):
                text_list[idx] = random.choice(string.ascii_letters)
            new_text = "".join(text_list)
        elif strategy == "truncate" and len(old_text) >= 3:
            cut = max(1, int(len(old_text) * random.uniform(0.3, 0.7)))
            new_text = old_text[:cut]
        elif strategy == "swap" and len(old_text) >= 2:
            text_list = list(old_text)
            i, j = random.sample(range(len(text_list)), 2)
            text_list[i], text_list[j] = text_list[j], text_list[i]
            new_text = "".join(text_list)
        else:
            text_list = list(old_text)
            text_list[random.randint(0, len(text_list) - 1)] = random.choice(string.ascii_letters)
            new_text = "".join(text_list)

        if new_text == old_text:
            new_text = old_text + "?"
        t.string = new_text
        return new_soup, f"Changed text of <{tag.name}> from '{old_text[:40]}' to '{new_text[:40]}'"

    # ── STYLE ────────────────────────────────────────────────────

    @staticmethod
    def perturb_style(soup: BeautifulSoup) -> Tuple[Optional[BeautifulSoup], str]:
        """Modify visual style attributes (font-family, opacity, border-radius, etc.)."""
        visible = _get_visible_elements(soup)
        if not visible:
            return None, ""

        strategies = []
        text_tags = [t for t in visible if t.name in
                     ("p", "span", "h1", "h2", "h3", "h4", "h5", "h6", "a",
                      "li", "td", "th", "label", "button", "div")]
        styled = [(t, _parse_style(t.get("style", ""))) for t in visible if t.get("style")]

        if text_tags:
            strategies.append("font_family")
        if styled:
            strategies.append("opacity")
            strategies.append("border_radius")
        if visible:
            strategies.append("text_decoration")
            strategies.append("box_shadow")

        if not strategies:
            return None, ""

        strategy = random.choice(strategies)
        new_soup = copy.deepcopy(soup)

        CSS_FONTS = ["Arial, sans-serif", "Georgia, serif", "Courier New, monospace",
                     "Verdana, sans-serif", "Times New Roman, serif", "Impact, sans-serif"]

        if strategy == "font_family":
            tag = random.choice(text_tags)
            old_val = _parse_style(tag.get("style", "")).get("font-family", "inherit")
            new_val = random.choice([f for f in CSS_FONTS if f != old_val])
            t = _find_matching_tag(new_soup, tag)
            if not t:
                return None, ""
            _set_style_prop(t, "font-family", new_val)
            return new_soup, f"Changed font-family of <{tag.name}> from '{old_val}' to '{new_val}'"

        elif strategy == "opacity":
            tag, style = random.choice(styled)
            old_val = style.get("opacity", "1")
            new_val = str(round(random.uniform(0.3, 0.8), 1))
            while new_val == old_val:
                new_val = str(round(random.uniform(0.3, 0.8), 1))
            t = _find_matching_tag(new_soup, tag)
            if not t:
                return None, ""
            _set_style_prop(t, "opacity", new_val)
            return new_soup, f"Changed opacity of <{tag.name}> from {old_val} to {new_val}"

        elif strategy == "border_radius":
            tag, style = random.choice(styled)
            old_val = style.get("border-radius", "0")
            new_val = f"{random.randint(4, 24)}px"
            t = _find_matching_tag(new_soup, tag)
            if not t:
                return None, ""
            _set_style_prop(t, "border-radius", new_val)
            return new_soup, f"Changed border-radius of <{tag.name}> from '{old_val}' to '{new_val}'"

        elif strategy == "text_decoration":
            tag = random.choice(visible)
            old_val = _parse_style(tag.get("style", "")).get("text-decoration", "none")
            opts = [o for o in ["underline", "line-through", "overline", "none"] if o != old_val]
            new_val = random.choice(opts)
            t = _find_matching_tag(new_soup, tag)
            if not t:
                return None, ""
            _set_style_prop(t, "text-decoration", new_val)
            return new_soup, f"Changed text-decoration of <{tag.name}> from '{old_val}' to '{new_val}'"

        elif strategy == "box_shadow":
            tag = random.choice(visible)
            old_val = _parse_style(tag.get("style", "")).get("box-shadow", "none")
            shadows = ["2px 2px 5px rgba(0,0,0,0.3)", "0 4px 8px rgba(0,0,0,0.2)",
                       "inset 0 0 10px rgba(0,0,0,0.15)", "none"]
            new_val = random.choice([s for s in shadows if s != old_val])
            t = _find_matching_tag(new_soup, tag)
            if not t:
                return None, ""
            _set_style_prop(t, "box-shadow", new_val)
            return new_soup, f"Changed box-shadow of <{tag.name}> from '{old_val}' to '{new_val}'"

        return None, ""


# ── Perturbation dispatch (ErrorCategory keys, matching svg_pipeline) ─

PERTURBATION_MAP = {
    ErrorCategory.COLOR: HTMLPerturbation.perturb_color,
    ErrorCategory.POSITION: HTMLPerturbation.perturb_position,
    ErrorCategory.SIZE: HTMLPerturbation.perturb_size,
    ErrorCategory.ELEMENT: HTMLPerturbation.perturb_element,
    ErrorCategory.TEXT: HTMLPerturbation.perturb_text,
    ErrorCategory.STYLE: HTMLPerturbation.perturb_style,
}

PERTURBATION_TYPES = list(PERTURBATION_MAP.keys())


# ── Synthetic HTML samples ───────────────────────────────────────────

def synthetic_htmls() -> List[str]:
    """Synthetic HTML pages with rich inline styles for perturbation testing."""
    return [
        # Card with button
        """<div style="padding: 20px; font-family: sans-serif; max-width: 400px;">
  <div style="background-color: #3498db; color: white; padding: 15px; border-radius: 8px;">
    <h2 style="margin: 0; font-size: 20px;">Card Title</h2>
    <p style="margin: 10px 0; font-size: 14px; line-height: 1.5;">This is a description of the card content with details.</p>
    <button style="background-color: #2ecc71; color: white; border: none; padding: 8px 16px; border-radius: 4px;">Click Me</button>
  </div>
</div>""",
        # Navigation bar
        """<nav style="background-color: #2c3e50; padding: 10px 20px; display: flex; justify-content: space-between; align-items: center;">
  <div style="color: #ecf0f1; font-size: 20px; font-weight: bold;">MyBrand</div>
  <div style="display: flex; gap: 15px;">
    <a style="color: #ecf0f1; text-decoration: none; font-size: 14px;">Home</a>
    <a style="color: #ecf0f1; text-decoration: none; font-size: 14px;">About</a>
    <a style="color: #ecf0f1; text-decoration: none; font-size: 14px;">Contact</a>
  </div>
</nav>""",
        # Form
        """<div style="padding: 20px; font-family: Arial; max-width: 300px;">
  <h3 style="color: #333; font-size: 22px;">Login Form</h3>
  <div style="margin-bottom: 10px;">
    <label style="display: block; margin-bottom: 5px; color: #555; font-size: 14px;">Username</label>
    <input type="text" style="width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 14px;" placeholder="Enter username">
  </div>
  <div style="margin-bottom: 10px;">
    <label style="display: block; margin-bottom: 5px; color: #555; font-size: 14px;">Password</label>
    <input type="password" style="width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 14px;" placeholder="Enter password">
  </div>
  <button style="background-color: #e74c3c; color: white; padding: 10px 20px; border: none; border-radius: 4px; width: 100%; font-size: 16px;">Sign In</button>
</div>""",
        # Table
        """<div style="padding: 20px; font-family: sans-serif;">
  <table style="border-collapse: collapse; width: 100%;">
    <tr style="background-color: #3498db; color: white;">
      <th style="padding: 10px; text-align: left; font-size: 14px;">Name</th>
      <th style="padding: 10px; text-align: left; font-size: 14px;">Score</th>
      <th style="padding: 10px; text-align: left; font-size: 14px;">Status</th>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="padding: 10px; color: #333; font-size: 14px;">Alice</td>
      <td style="padding: 10px; font-size: 14px;">95</td>
      <td style="padding: 10px; color: #2ecc71; font-size: 14px;">Pass</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="padding: 10px; color: #333; font-size: 14px;">Bob</td>
      <td style="padding: 10px; font-size: 14px;">67</td>
      <td style="padding: 10px; color: #e74c3c; font-size: 14px;">Fail</td>
    </tr>
  </table>
</div>""",
        # Hero section
        """<div style="background-color: #1a1a2e; color: white; padding: 60px 40px; text-align: center; font-family: Georgia;">
  <h1 style="font-size: 36px; margin-bottom: 20px;">Welcome to Our Platform</h1>
  <p style="font-size: 18px; color: #aaa; max-width: 600px; margin: 0 auto;">Build amazing things with powerful tools and resources.</p>
  <button style="background-color: #e94560; color: white; border: none; padding: 12px 30px; font-size: 16px; border-radius: 25px; margin-top: 30px;">Get Started</button>
</div>""",
        # Pricing cards
        """<div style="display: flex; gap: 20px; padding: 20px; font-family: sans-serif;">
  <div style="flex: 1; border: 2px solid #3498db; border-radius: 8px; padding: 20px; text-align: center;">
    <h3 style="color: #3498db; font-size: 18px;">Basic</h3>
    <p style="font-size: 24px; font-weight: bold; color: #333;">$9/mo</p>
    <p style="color: #777; font-size: 14px;">5 projects</p>
    <button style="background-color: #3498db; color: white; border: none; padding: 8px 20px; border-radius: 4px; font-size: 14px;">Choose</button>
  </div>
  <div style="flex: 1; border: 2px solid #e74c3c; border-radius: 8px; padding: 20px; text-align: center;">
    <h3 style="color: #e74c3c; font-size: 18px;">Pro</h3>
    <p style="font-size: 24px; font-weight: bold; color: #333;">$29/mo</p>
    <p style="color: #777; font-size: 14px;">Unlimited projects</p>
    <button style="background-color: #e74c3c; color: white; border: none; padding: 8px 20px; border-radius: 4px; font-size: 14px;">Choose</button>
  </div>
</div>""",
        # Image gallery
        """<div style="display: flex; gap: 10px; padding: 20px; flex-wrap: wrap;">
  <img src="https://via.placeholder.com/150" style="width: 150px; height: 150px; border-radius: 8px;">
  <img src="https://via.placeholder.com/150" style="width: 150px; height: 150px; border-radius: 8px;">
  <div style="width: 150px; height: 150px; background-color: #f39c12; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: white; font-size: 14px;">+12 more</div>
</div>""",
    ]


# ── Pipeline ─────────────────────────────────────────────────────────

class HTMLPipeline:
    """End-to-end pipeline: load HTML -> perturb -> render -> output."""

    def __init__(self, render_width=1280, render_height=720, seed=None):
        self.render_width = render_width
        self.render_height = render_height
        if seed is not None:
            random.seed(seed)

    def load_webcode2m(self, max_samples=None) -> List[str]:
        try:
            from datasets import load_dataset
        except ImportError:
            print("[WARN] `datasets` not installed.")
            return []
        try:
            ds = load_dataset("nickssonfreitas/WebCode-2M", split="train", streaming=True)
            htmls = []
            for i, sample in enumerate(ds):
                if max_samples and i >= max_samples:
                    break
                html_str = sample.get("html") or sample.get("code") or sample.get("text") or ""
                if html_str and "<" in html_str:
                    htmls.append(html_str)
            if htmls:
                print(f"[INFO] Loaded {len(htmls)} HTMLs from WebCode2M")
            return htmls
        except Exception as e:
            print(f"[WARN] Failed to load WebCode2M: {e}")
            return []

    def process_one(self, html_code, error_category=None):
        soup = BeautifulSoup(html_code, "html.parser")
        if error_category is None:
            error_category = random.choice(PERTURBATION_TYPES)

        perturb_fn = PERTURBATION_MAP[error_category]
        result = perturb_fn(soup)
        if result is None:
            return None
        new_soup, description = result
        if new_soup is None:
            return None

        original_html = html_code
        perturbed_html = str(new_soup)
        # Normalize whitespace before comparing to avoid BS4 serialization artifacts
        _ws = lambda s: re.sub(r'\s+', ' ', s).strip()
        if _ws(original_html) == _ws(perturbed_html):
            return None

        try:
            original_img = render_html_sync(original_html, self.render_width, self.render_height)
            perturbed_img = render_html_sync(perturbed_html, self.render_width, self.render_height)
        except Exception as e:
            print(f"[WARN] Render failed: {e}")
            return None

        return {
            "original_html": original_html,
            "perturbed_html": perturbed_html,
            "original_img": original_img,
            "perturbed_img": perturbed_img,
            "error_type": error_category.value,
            "error_description": description,
        }

    def run(self, num_samples=100, output_dir=None, use_hf=True):
        htmls = []
        if use_hf:
            htmls = self.load_webcode2m(max_samples=num_samples * 2)
        if not htmls:
            print("[INFO] Using synthetic HTMLs.")
            htmls = synthetic_htmls()

        results = []
        attempts = 0
        max_attempts = num_samples * 5

        while len(results) < num_samples and attempts < max_attempts:
            html = random.choice(htmls)
            sample = self.process_one(html)
            if sample is not None:
                results.append(sample)
                if len(results) % 10 == 0:
                    print(f"[INFO] Progress: {len(results)}/{num_samples}")
            attempts += 1

        print(f"[INFO] Generated {len(results)}/{num_samples} ({attempts} attempts)")
        if output_dir:
            self._save(results, output_dir)
        return results

    def _save(self, results, output_dir):
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
                "original_html": sample["original_html"],
                "perturbed_html": sample["perturbed_html"],
                "original_img_path": str(orig_path),
                "perturbed_img_path": str(pert_path),
            })

        with open(out / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Saved {len(results)} samples to {output_dir}")


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HTML perturbation pipeline.")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="data/processed/html/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no_hf", action="store_true")
    args = parser.parse_args()

    pipeline = HTMLPipeline(render_width=args.width, render_height=args.height, seed=args.seed)
    pipeline.run(num_samples=args.num_samples, output_dir=args.output_dir, use_hf=not args.no_hf)


if __name__ == "__main__":
    main()
