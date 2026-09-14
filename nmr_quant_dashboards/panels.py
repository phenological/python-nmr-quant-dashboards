"""Panel builders: draw one fit, its residual, quality badges, a median overlay,
or a full-spectrum overview.

Every function draws into an Axes the caller supplies and returns it; nothing is
written to disk and ``plt.show`` is never called. Inputs are plain arrays (see
:class:`FitPanel`), so this package depends only on numpy and matplotlib (scipy
is used by :mod:`nmr_quant_dashboards.contours`).
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
    overlays; ``metrics``, ``quality`` and ``failed`` drive the caption, colour
    and badges.
    """

    x: np.ndarray
    y: np.ndarray
    y_model: np.ndarray | None = None
    components: list[tuple[str, np.ndarray]] | None = None
    baseline: np.ndarray | None = None
    label: str = ""
    quality: bool | None = None
    metrics: dict | None = None
    failed: list[str] | None = None
    reference: FitPanel | None = None
    median: np.ndarray | None = None
    xlim: tuple[float, float] | None = None


# ── fit decomposition ─────────────────────────────────────────────────────────
def draw_fit_panel(ax, panel: FitPanel, *, xlim=None, fill=True, invert=False,
                   show_components=True, show_baseline=True, show_legend=False,
                   show_metrics=True):
    """Draw raw data, the fitted model (optionally filled), components/baseline.

    Returns ``ax``. On bad/empty input, renders an ERROR title instead of
    raising, so one broken panel does not sink a whole dashboard.
    """
    try:
        x = np.asarray(panel.x, dtype=float)
        y = np.asarray(panel.y, dtype=float)
        if x.size == 0 or y.size == 0:
            raise ValueError("empty data")
        base = (np.asarray(panel.baseline, dtype=float)
                if panel.baseline is not None else np.zeros_like(x))

        if panel.y_model is not None:
            ym = np.asarray(panel.y_model, dtype=float)
            r2 = (panel.metrics or {}).get("r2")
            lbl = f"fit (R²={r2:.4f})" if r2 is not None else "fit"
            if fill:
                ax.fill_between(x, base, ym, alpha=0.22, color=style.MODEL,
                                zorder=1, label=lbl)
                ax.plot(x, ym, color=style.MODEL, lw=1.0, zorder=3)
            else:
                ax.plot(x, ym, color=style.MODEL, lw=1.2, zorder=3, label=lbl)
        if show_components and panel.components:
            for name, comp in panel.components:
                col = style.SATELLITE if str(name).lower().startswith("sat") else style.COMPONENT
                ax.plot(x, np.asarray(comp, dtype=float), color=col, lw=1.0,
                        ls="--", alpha=0.9, zorder=3, label=name)
        if show_baseline and panel.baseline is not None:
            ax.plot(x, base, color=style.BASELINE, lw=1.0, ls="--", zorder=2,
                    label="baseline")
        ax.plot(x, y, color=style.DATA, lw=1.1, zorder=4, label="data")

        color = style.quality_color(panel.quality)
        title = panel.label or ""
        if show_metrics:
            cap = style.metrics_caption(panel.metrics)
            title = f"{title}   {cap}" if (title and cap) else (title or cap)
        ax.set_title(title, fontsize=8, fontweight="bold", color=color)
        ax.set_ylabel("Intensity", fontsize=7)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=6)

        lim = xlim if xlim is not None else panel.xlim
        if lim is not None:
            ax.set_xlim(*lim)
        elif invert:
            ax.invert_xaxis()
        if show_legend:
            ax.legend(fontsize=6, loc="upper right")
    except Exception as exc:  # noqa: BLE001 - a broken panel must not sink the figure
        ax.set_title(f"ERROR: {exc}", fontsize=8, color=style.FAIL)
        ax.axis("off")
    return ax


# ── residual strip ────────────────────────────────────────────────────────────
def draw_residual(ax, panel: FitPanel, *, xlim=None, as_percent=False,
                  peak_height=None, show_area=False, invert=False):
    """Draw the residual (data - model) with a zero reference line.

    With ``as_percent`` and a ``peak_height`` (defaults to the panel's fitted
    height), the residual is shown as a percentage of peak height; with
    ``show_area`` the signed residual area on each side of the centre is shaded
    (as in the fovea residual panel).
    """
    x = np.asarray(panel.x, dtype=float)
    if panel.y_model is None:
        ax.text(0.5, 0.5, "no model", transform=ax.transAxes,
                ha="center", va="center", color=style.GREY)
        return ax
    resid = np.asarray(panel.y, dtype=float) - np.asarray(panel.y_model, dtype=float)

    h = peak_height if peak_height is not None else (panel.metrics or {}).get("height")
    if as_percent and h:
        resid = resid / h * 100.0
        ax.set_ylabel("resid (% peak)", fontsize=6)
    else:
        ax.set_ylabel("resid", fontsize=6)

    if show_area:
        _draw_residual_area(ax, x, resid, panel)

    ax.axhline(0.0, color=style.GREY, lw=0.8, ls="--")
    ax.plot(x, resid, color=style.RESIDUAL, lw=0.9, zorder=2)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=6)
    lim = xlim if xlim is not None else panel.xlim
    if lim is not None:
        ax.set_xlim(*lim)
    elif invert:
        ax.invert_xaxis()
    return ax


def _draw_residual_area(ax, x, resid, panel, scale=10):
    """Signed residual-area shading, split left/right of the fitted centre."""
    c = float((panel.metrics or {}).get("center", x[int(np.argmax(np.abs(resid)))]))
    trapz = getattr(np, "trapezoid", None) or np.trapz

    def _rect(mask):
        if not mask.any():
            return 0.0, 0.0, c
        span = abs(float(x[mask].max() if x[mask].max() != c else c) - c) or 1.0
        up = float(trapz(np.maximum(resid[mask], 0), x[mask])) / span
        dn = float(trapz(np.minimum(resid[mask], 0), x[mask])) / span
        return up, dn, float(x[mask].max() if (x[mask] >= c).any() else x[mask].min())

    m_pos, m_neg = x >= c, x <= c
    up_p, dn_p, x_l = _rect(m_pos)
    up_n, dn_n, x_r = _rect(m_neg)
    for (x0, x1, col, up, dn) in [(c, x_l, style.MODEL, up_p, dn_p),
                                  (x_r, c, style.BASELINE, up_n, dn_n)]:
        ax.fill_between([x0, x1], 0, up * scale, alpha=0.35, color=col, zorder=0)
        ax.fill_between([x0, x1], 0, dn * scale, alpha=0.35, color=col, zorder=0)


# ── quality badges ────────────────────────────────────────────────────────────
_BADGES = [
    ("kde_score", "kde_score", "kde", "{:.3f}"),
    ("fwhm", "fwhm", "fwhm", "{:.5f}"),
    ("area_total", "area", "area", "{:.4f}"),
    ("center", "center", "center", "{:+.5f} ppm"),
    ("r2", "r2", "r2", "{:.4f}"),
    ("r2_adj", "r2_adj", "r²_adj", "{:.4f}"),
    ("srr", "srr", "SRR", "{:.1f}"),
]


def draw_badges(ax, metrics: dict, *, failed=None, r2_threshold=0.99,
                x=0.005, y0=0.97, dy=0.13):
    """Stack PASS/FAIL badges (kde, fwhm, area, center, r2, r2_adj, srr) top-left.

    ``failed`` is the set/list of criterion names that failed the quality map
    (from a ``quality_reason`` string). ``failed=None`` means no quality map was
    applied — everything shows PASS except ``r2``, which uses ``r2_threshold``.
    """
    has_quality = failed is not None
    failed = set(failed or ())
    yp = y0
    for metric_key, failed_key, label, fmt in _BADGES:
        v = (metrics or {}).get(metric_key)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            continue
        if metric_key == "r2" and not has_quality:
            passed = v >= r2_threshold
        else:
            passed = (failed_key not in failed) if has_quality else True
        col = style.PASS if passed else style.FAIL
        mark = "✓" if passed else "✗"
        ax.text(x, yp, f"{mark} {'PASS' if passed else 'FAIL'}  {label}={fmt.format(v)}",
                transform=ax.transAxes, fontsize=8, fontweight="bold", color="white",
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=col, edgecolor="none",
                          alpha=0.85))
        yp -= dy
    return ax


# ── median overlay ────────────────────────────────────────────────────────────
def draw_median(ax, x, y, median, *, xlim=None, invert=False, show_legend=True):
    """Overlay the cohort median spectrum with this spectrum for context."""
    x = np.asarray(x, dtype=float)
    ax.plot(x, np.asarray(median, dtype=float), color=style.DATA, lw=0.9, label="median")
    ax.plot(x, np.asarray(y, dtype=float), color=style.MEDIAN, lw=0.7, alpha=0.7,
            label="this spectrum")
    ax.set_ylabel("Intensity", fontsize=7)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=6)
    if xlim is not None:
        ax.set_xlim(*xlim)
    elif invert:
        ax.invert_xaxis()
    if show_legend:
        ax.legend(fontsize=6, loc="upper right")
    return ax


# ── full-spectrum overview ────────────────────────────────────────────────────
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
