"""Shared colours and small drawing helpers.

Colour semantics live in one place so panels and layouts stay consistent:
pass = green, fail = red, no verdict = blue/grey.
"""

from __future__ import annotations

PASS = "#2ecc71"
FAIL = "#e74c3c"
NEUTRAL = "#3498db"
GREY = "#95a5a6"

DATA = "#222222"
MODEL = "#e67e22"
COMPONENT = "#3498db"
BASELINE = "#7f8c8d"
RESIDUAL = "#7f8c8d"


def quality_color(quality: bool | None) -> str:
    """Map a pass/fail/None verdict to its colour."""
    if quality is True:
        return PASS
    if quality is False:
        return FAIL
    return NEUTRAL


def metrics_caption(metrics: dict | None) -> str:
    """Short one-line caption from a metrics dict (R2 / SRR if present)."""
    if not metrics:
        return ""
    bits = []
    if "r2" in metrics:
        bits.append(f"R²={metrics['r2']:.4f}")
    if "srr" in metrics:
        bits.append(f"SRR={metrics['srr']:.0f}")
    if "fwhm" in metrics:
        bits.append(f"FWHM={metrics['fwhm']:.4f}")
    return "  ".join(bits)
