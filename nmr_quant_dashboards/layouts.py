"""Opinionated multi-panel dashboard figures built from the panel builders."""

from __future__ import annotations

import math

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from .panels import (
    FitPanel,
    draw_badges,
    draw_fit_panel,
    draw_overview,
    draw_residual,
)


def dashboard_full(panels: list[FitPanel], *, overview=None, title="",
                   figsize=None, show_badges=False):
    """A full dashboard: optional overview row, then per-signal fit + residual.

    ``overview`` is ``(full_x, full_y, rois[, folder_label])`` or None; see
    :func:`nmr_quant_dashboards.panels.draw_overview` for the ``rois`` shape.
    Returns a Figure, or ``None`` when ``panels`` is empty.
    """
    if not panels:
        return None
    n = len(panels)
    has_ov = overview is not None
    nrows = (1 if has_ov else 0) + 2 * n
    height_ratios = ([2] if has_ov else []) + [3, 1] * n

    fig = plt.figure(figsize=figsize or (12, (2.5 if has_ov else 0) + 3.0 * n))
    gs = gridspec.GridSpec(nrows, 1, figure=fig, height_ratios=height_ratios,
                           hspace=0.55)

    row0 = 0
    if has_ov:
        full_x, full_y, rois, *rest = overview
        folder_label = rest[0] if rest else ""
        draw_overview(fig.add_subplot(gs[0, 0]), full_x, full_y, rois,
                      folder_label=folder_label)
        row0 = 1

    for i, panel in enumerate(panels):
        ax_fit = fig.add_subplot(gs[row0 + 2 * i, 0])
        draw_fit_panel(ax_fit, panel)
        if show_badges and panel.metrics:
            draw_badges(ax_fit, panel.metrics, failed=panel.failed)
        draw_residual(fig.add_subplot(gs[row0 + 2 * i + 1, 0]), panel)

    if title:
        fig.suptitle(title, fontsize=10, fontweight="bold")
    return fig


def dashboard_compact(panels: list[FitPanel], *, title="", ncols=2, figsize=None):
    """A compact grid of fit panels only (no overview, no residual strips).

    Returns a Figure, or ``None`` when ``panels`` is empty.
    """
    if not panels:
        return None
    n = len(panels)
    nrows = math.ceil(n / ncols)
    fig = plt.figure(figsize=figsize or (6.0 * ncols, 2.4 * nrows))
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.55, wspace=0.3)
    for i, panel in enumerate(panels):
        draw_fit_panel(fig.add_subplot(gs[i // ncols, i % ncols]), panel)
    if title:
        fig.suptitle(title, fontsize=10, fontweight="bold")
    return fig
