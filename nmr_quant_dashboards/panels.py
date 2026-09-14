"""Panel builders: draw one fit, its residual, or a full-spectrum overview.

Every function draws into an Axes the caller supplies and returns it; nothing
is written to disk and ``plt.show`` is never called. Inputs are plain arrays
(see :class:`FitPanel`), so this package depends on nothing but numpy and
matplotlib.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from . import style


@dataclass
class FitPanel:
    """Everything needed to draw one signal's fit, as pre-computed arrays.

    The caller (e.g. fovea, via ``QuantResults.model_curve``) evaluates the
    model; this package only plots. ``components`` and ``baseline`` are optional
    overlays; ``metrics`` and ``quality`` drive the caption and colour.
    """

    x: np.ndarray
    y: np.ndarray
    y_model: np.ndarray | None = None
    components: list[tuple[str, np.ndarray]] | None = None
    baseline: np.ndarray | None = None
    label: str = ""
    quality: bool | None = None
    metrics: dict | None = None
    reference: FitPanel | None = None
    xlim: tuple[float, float] | None = None


def draw_fit_panel(ax, panel: FitPanel, *, xlim=None, show_components=True,
                   show_baseline=True, show_legend=False, show_metrics=True):
    """Draw raw data, the fitted model, and optional components/baseline.

    Returns ``ax``. On bad/empty input, renders an ERROR title instead of
    raising, so one broken panel does not sink a whole dashboard.
    """
    try:
        x = np.asarray(panel.x, dtype=float)
        y = np.asarray(panel.y, dtype=float)
        if x.size == 0 or y.size == 0:
            raise ValueError("empty data")

        ax.plot(x, y, color=style.DATA, lw=0.9, label="data")
        if panel.y_model is not None:
            ax.plot(x, np.asarray(panel.y_model, dtype=float),
                    color=style.MODEL, lw=1.2, label="fit")
        if show_components and panel.components:
            for name, comp in panel.components:
                ax.plot(x, np.asarray(comp, dtype=float), color=style.COMPONENT,
                        lw=0.7, alpha=0.7, label=name)
        if show_baseline and panel.baseline is not None:
            ax.plot(x, np.asarray(panel.baseline, dtype=float),
                    color=style.BASELINE, lw=0.8, ls="--", label="baseline")

        color = style.quality_color(panel.quality)
        title = panel.label or ""
        if show_metrics:
            cap = style.metrics_caption(panel.metrics)
            if cap:
                title = f"{title}   {cap}" if title else cap
        ax.set_title(title, fontsize=8, fontweight="bold", color=color)

        lim = xlim if xlim is not None else panel.xlim
        if lim is not None:
            ax.set_xlim(*lim)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)
        if show_legend:
            ax.legend(fontsize=6, loc="best")
    except Exception as exc:  # noqa: BLE001 - a broken panel must not sink the figure
        ax.set_title(f"ERROR: {exc}", fontsize=8, color=style.FAIL)
        ax.axis("off")
    return ax


def draw_residual(ax, panel: FitPanel, *, xlim=None):
    """Draw the residual (data - model) with a zero reference line."""
    x = np.asarray(panel.x, dtype=float)
    if panel.y_model is None:
        ax.text(0.5, 0.5, "no model", transform=ax.transAxes,
                ha="center", va="center", color=style.GREY)
        return ax
    resid = np.asarray(panel.y, dtype=float) - np.asarray(panel.y_model, dtype=float)
    ax.axhline(0.0, color=style.GREY, lw=0.6)
    ax.plot(x, resid, color=style.RESIDUAL, lw=0.7)
    lim = xlim if xlim is not None else panel.xlim
    if lim is not None:
        ax.set_xlim(*lim)
    ax.tick_params(labelsize=6)
    ax.set_ylabel("resid", fontsize=6)
    return ax


def draw_overview(ax, full_x, full_y, rois: Sequence[tuple], *, folder_label=""):
    """Draw a full spectrum with fitted ROIs shaded by quality.

    ``rois`` is a sequence of ``(x_lo, x_hi, label, quality)``. If ``full_y`` is
    None, a placeholder is drawn instead of raising.
    """
    if full_y is None or full_x is None:
        ax.text(0.5, 0.5, "Full spectrum not available", transform=ax.transAxes,
                ha="center", va="center", color=style.GREY)
        ax.set_title(folder_label, fontsize=7, fontweight="bold")
        return ax

    full_x = np.asarray(full_x, dtype=float)
    full_y = np.asarray(full_y, dtype=float)
    ax.plot(full_x, full_y, color=style.DATA, lw=0.6, alpha=0.85)
    top = float(np.nanmax(full_y)) if full_y.size else 1.0
    for x_lo, x_hi, label, quality in rois:
        color = style.quality_color(quality)
        ax.axvspan(x_lo, x_hi, alpha=0.18, color=color, zorder=0)
        ax.text((x_lo + x_hi) / 2, top * 1.01, label, ha="center", va="bottom",
                fontsize=5.5, color=color, rotation=60, clip_on=True)
    ax.set_title(folder_label, fontsize=7, fontweight="bold")
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.2)
    return ax
