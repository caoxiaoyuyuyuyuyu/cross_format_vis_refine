"""Unified error taxonomy for cross-format code refinement."""

from enum import Enum
from typing import Dict, List


class ErrorCategory(Enum):
    """6 error categories for visual code refinement."""
    COLOR = "color"
    POSITION = "position"
    SIZE = "size"
    ELEMENT = "element"
    TEXT = "text"
    STYLE = "style"


# SVG-specific error subtypes
SVG_ERROR_SUBTYPES: Dict[ErrorCategory, List[str]] = {
    ErrorCategory.COLOR: ["fill", "stroke", "gradient"],
    ErrorCategory.POSITION: ["translate", "cx_cy", "x_y"],
    ErrorCategory.SIZE: ["scale", "width_height", "radius"],
    ErrorCategory.ELEMENT: ["delete", "duplicate", "reorder"],
    ErrorCategory.TEXT: ["char_replace", "truncate", "swap"],
    ErrorCategory.STYLE: ["font_family", "stroke_dasharray", "opacity",
                          "fill_rule", "stroke_width"],
}

# HTML-specific error subtypes
HTML_ERROR_SUBTYPES: Dict[ErrorCategory, List[str]] = {
    ErrorCategory.COLOR: ["background-color", "color", "border-color"],
    ErrorCategory.POSITION: ["margin", "padding", "flex-direction", "text-align"],
    ErrorCategory.SIZE: ["width", "height", "font-size", "border-width"],
    ErrorCategory.ELEMENT: ["delete", "replace"],
    ErrorCategory.TEXT: ["typo", "truncate", "swap"],
    ErrorCategory.STYLE: ["font_family", "opacity", "border_radius",
                          "text_decoration", "box_shadow"],
}
