"""nmr-quant-dashboards — matplotlib dashboards for NMR fit inspection and QA.

Panels take pre-computed arrays (:class:`FitPanel`), so the drawing primitives
depend only on numpy and matplotlib; :func:`fit_quality_contours` additionally
uses scipy. The caller supplies the fitted model curve (e.g. via
``nmr_quant.QuantResults.model_curve``); this package only draws.

Building blocks (compose your own layout):
    FitPanel, draw_fit_panel, draw_residual, draw_badges, draw_median,
    draw_overview
Ready-made layouts:
    dashboard_full, dashboard_compact
Fit-quality space:
    fit_quality_contours
"""

from __future__ import annotations

from .contours import fit_quality_contours
from .layouts import dashboard_compact, dashboard_full
from .panels import (
    FitPanel,
    draw_badges,
    draw_fit_panel,
    draw_median,
    draw_overview,
    draw_residual,
)

__all__ = [
    "FitPanel",
    "dashboard_compact",
    "dashboard_full",
    "draw_badges",
    "draw_fit_panel",
    "draw_median",
    "draw_overview",
    "draw_residual",
    "fit_quality_contours",
]
