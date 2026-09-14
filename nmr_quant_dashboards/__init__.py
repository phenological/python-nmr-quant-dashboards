"""nmr-quant-dashboards — matplotlib dashboards for NMR fit inspection and QA.

Panels take pre-computed arrays (:class:`FitPanel`), so this package depends on
nothing but numpy and matplotlib. The caller supplies the fitted model curve
(e.g. via ``nmr_quant.QuantResults.model_curve``); this package only draws.

Public API:
    FitPanel                    the plain-array input for one signal's fit
    draw_fit_panel              data + model (+ components/baseline) into an Axes
    draw_residual               residual strip into an Axes
    draw_overview               full spectrum with ROIs shaded by quality
    dashboard_full              overview + per-signal fit/residual figure
    dashboard_compact           compact grid of fit panels
"""

from __future__ import annotations

from .layouts import dashboard_compact, dashboard_full
from .panels import FitPanel, draw_fit_panel, draw_overview, draw_residual

__all__ = [
    "FitPanel",
    "dashboard_compact",
    "dashboard_full",
    "draw_fit_panel",
    "draw_overview",
    "draw_residual",
]
