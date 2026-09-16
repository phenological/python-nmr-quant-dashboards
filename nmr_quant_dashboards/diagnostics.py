"""VLM-oriented diagnostic building blocks (exploration branch).

Small, composable pieces that make a peak fit legible to a vision model. Each
does one thing; layouts compose them. Salvaged ideas:

- **baseline-subtracted amplitudes** — plot ``exp - baseline`` so every height
  is measured from y = 0 (no mental baseline arithmetic).
- **explicit total fit** — draw the summed model, so there is no sum rule to do.
- **filled components** — each component is a filled area; it survives
  downscaling and a zero-amplitude component is literally empty.
- **robust noise band** — ``N_pp`` from the residual MAD, shaded as a band, so
  "above noise" becomes a spatial, binary judgement immune to unmodelled peaks.
- **residual as a shape** — flat inside the band = good fit.
- **direct labels** — annotate components in place, no legend to bind.

These take plain arrays (like the mockup) so they are easy to reason about; a
later step wires them to ``FitPanel``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

C_EXP = "#444444"   # experimental trace
C_TOTAL = "#c0392b"  # total fit / residual
BAND = "0.80"       # noise-band grey
ZERO = "0.35"       # zero line grey


# ── noise ─────────────────────────────────────────────────────────────────────
def robust_noise_pp(residual, k: float = 6.0) -> float:
    """Peak-to-peak noise from the residual via MAD (Gaussian-consistent).

    ``N_pp = k * 1.4826 * MAD(residual)``. Region-wide and immune to unmodelled
    peaks, so it needs no clean off-peak window inside the plot.
    """
    r = np.asarray(residual, dtype=float)
    mad = float(np.median(np.abs(r - np.median(r))))
    return float(k * 1.4826 * mad)


def draw_noise_band(ax, n_pp: float, *, zero: bool = True) -> None:
    """Shade +/- n_pp/2 and (optionally) draw the zero line."""
    ax.axhspan(-n_pp / 2, n_pp / 2, color=BAND, zorder=0)
    if zero:
        ax.axhline(0, color=ZERO, lw=1.0, ls="--", zorder=1)


# ── overlay (baseline-subtracted, filled components, total fit) ────────────────
def draw_overlay(ax, ppm, exp_sub, total, components: dict, *, colors: dict,
                 n_pp: float | None = None, labels: bool = True,
                 label_at: dict | None = None):
    """Baseline-subtracted overlay: filled components + experimental + total fit.

    ``components`` maps name -> curve; ``colors`` maps name -> colour. When
    ``labels`` and ``label_at`` (name -> (x, y)) are given, components are
    annotated in place (no legend).
    """
    ppm = np.asarray(ppm, dtype=float)
    if n_pp is not None:
        draw_noise_band(ax, n_pp)
    for name, curve in components.items():
        col = colors.get(name, C_EXP)
        ax.fill_between(ppm, 0, curve, color=col, alpha=0.38, lw=0, zorder=2)
        ax.plot(ppm, curve, color=col, lw=1.6, zorder=3)
    ax.plot(ppm, np.asarray(exp_sub, float), color=C_EXP, lw=1.3, zorder=4)
    ax.plot(ppm, np.asarray(total, float), color=C_TOTAL, lw=2.0, zorder=5)
    ax.set_xlim(ppm.max(), ppm.min())
    if labels and label_at:
        for name, (x, y) in label_at.items():
            ax.text(x, y, name, color=colors.get(name, C_EXP),
                    fontweight="bold", ha="right", va="center", fontsize=9)
    return ax


# ── residual as a shape ────────────────────────────────────────────────────────
def draw_residual_band(ax, ppm, residual, n_pp: float, *, annotate=()):
    """Residual with the noise band; flat inside the band = good fit.

    ``annotate`` is an iterable of ``(x, label)`` markers (e.g. unmodelled peaks).
    """
    ppm = np.asarray(ppm, dtype=float)
    residual = np.asarray(residual, dtype=float)
    draw_noise_band(ax, n_pp)
    ax.plot(ppm, residual, color=C_TOTAL, lw=1.3, zorder=3)
    ax.set_xlim(ppm.max(), ppm.min())
    ax.set_ylim(-2.2 * n_pp, 2.2 * n_pp)
    for x, lab in annotate:
        i = int(np.argmin(np.abs(ppm - x)))
        ax.annotate(lab, xy=(x, residual[i]), xytext=(x, 1.6 * n_pp), ha="center",
                    fontsize=9, color="0.25",
                    arrowprops=dict(arrowstyle="->", color="0.35", lw=1.0))
    return ax


# ── per-component readout + solo panel ─────────────────────────────────────────
def component_rows(ppm, exp_sub, component_curve, centers, n_pp: float) -> list[dict]:
    """One readout row per line: ppm, fit height, exp height, rho, SNR.

    ``rho = h_exp / h_fit`` (1.0 = the fit matches the data at the apex);
    ``SNR = h_exp / N_pp``.
    """
    ppm = np.asarray(ppm, dtype=float)
    exp_sub = np.asarray(exp_sub, dtype=float)
    component_curve = np.asarray(component_curve, dtype=float)
    rows = []
    for x0 in centers:
        i = int(np.argmin(np.abs(ppm - x0)))
        h_fit, h_exp = float(component_curve[i]), float(exp_sub[i])
        rows.append(dict(ppm=float(x0), h_fit=h_fit, h_exp=h_exp,
                         rho=(h_exp / h_fit if h_fit else np.nan),
                         snr=(h_exp / n_pp if n_pp else np.nan)))
    return rows


def _k(v: float) -> str:
    return f"{v/1000:.1f}k"


def draw_component_panel(ax, ppm, solo, component_curve, centers, *, color,
                         rows: list[dict], n_pp: float, title: str = ""):
    """One metabolite panel: the neighbour-removed experimental trace vs this
    component, filled, with dashed line centres and a numeric readout box.

    ``solo`` is the experimental trace with the *other* components' fits removed,
    so this component can be compared to the data directly (the sum rule done in
    code, not in the reader's head).
    """
    ppm = np.asarray(ppm, dtype=float)
    lo, hi = min(centers) - 0.004, max(centers) + 0.004
    sel = (ppm >= lo) & (ppm <= hi)
    draw_noise_band(ax, n_pp)
    ax.fill_between(ppm[sel], 0, np.asarray(component_curve, float)[sel],
                    color=color, alpha=0.38, lw=0, zorder=3)
    ax.plot(ppm[sel], np.asarray(component_curve, float)[sel], color=color, lw=2.2, zorder=4)
    ax.plot(ppm[sel], np.asarray(solo, float)[sel], color=C_EXP, lw=1.3, zorder=5)
    for x0 in centers:
        ax.axvline(x0, color=color, lw=0.9, ls="--", alpha=0.7, zorder=1)
    ax.set_xlim(hi, lo)
    top = max(np.asarray(component_curve, float)[sel].max(),
              np.asarray(solo, float)[sel].max()) * 1.75
    ax.set_ylim(-1.6 * n_pp, top)
    if title:
        ax.set_title(title, color=color, fontweight="bold", pad=18)
        ax.text(0.5, 1.015, "neighbour's fit removed from the experimental trace",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5,
                color="0.35")
    txt = "\n".join(
        f"{r['ppm']:.4f}  fit {_k(r['h_fit'])}  exp {_k(r['h_exp'])}  "
        f"r {r['rho']:.2f}  SNR {r['snr']:.1f}" for r in rows)
    ax.text(0.5, 0.965, txt, transform=ax.transAxes, ha="center", va="top",
            fontsize=7.6, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", lw=0.8))
    return ax


# ── standalone figures (one artifact per module) ───────────────────────────────
# Each draw_* renders into a caller-supplied Axes so a composite dashboard can
# assemble them for humans; these *_figure helpers wrap one module into its own
# Figure so each panel can be saved / sent to the vision model individually.
def overlay_figure(ppm, exp_sub, total, components, *, colors, n_pp=None,
                   label_at=None, folder_label="", figsize=(8.0, 3.4)):
    fig, ax = plt.subplots(figsize=figsize)
    draw_overlay(ax, ppm, exp_sub, total, components, colors=colors, n_pp=n_pp,
                 label_at=label_at)
    ax.set_ylabel("intensity - baseline")
    ax.set_xlabel("ppm")
    band = f"  |  N(pp)={_k(n_pp)}" if n_pp else ""
    ax.set_title(f"{folder_label}  experimental vs total fit (components shaded){band}",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def residual_figure(ppm, residual, n_pp, *, annotate=(), folder_label="",
                    figsize=(8.0, 2.3)):
    fig, ax = plt.subplots(figsize=figsize)
    draw_residual_band(ax, ppm, residual, n_pp, annotate=annotate)
    ax.set_ylabel("residual")
    ax.set_xlabel("ppm")
    ax.set_title(f"{folder_label}  residual (flat inside the band = good fit)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def component_figure(ppm, solo, component_curve, centers, *, color, rows, n_pp,
                     title="", figsize=(5.0, 4.0)):
    fig, ax = plt.subplots(figsize=figsize)
    draw_component_panel(ax, ppm, solo, component_curve, centers, color=color,
                         rows=rows, n_pp=n_pp, title=title)
    ax.set_ylabel("intensity - baseline")
    ax.set_xlabel("ppm")
    fig.tight_layout()
    return fig


# ── montage (reconstruct a dashboard from saved panel PNGs) ─────────────────────
def montage(rows, *, width: float = 8.0, dpi: int = 100, space: float = 0.03):
    """Stitch saved panel PNGs into one image for human review.

    ``rows`` is a list of rows, each a list of image paths laid out left-to-right
    (e.g. ``[[overlay], [residual], [metabolite_a, metabolite_b]]``). Images are
    shown at their own aspect ratio, so the montage is exactly the panels the
    vision model saw, reassembled — no re-rendering.
    """
    import math
    from functools import reduce

    import matplotlib.image as mpimg

    imgs = [[mpimg.imread(p) for p in row] for row in rows]
    ncols = reduce(math.lcm, (len(r) for r in imgs))
    row_h = [(width / len(r)) * (r[0].shape[0] / r[0].shape[1]) for r in imgs]

    fig = plt.figure(figsize=(width, sum(row_h) * (1 + space)), dpi=dpi)
    gs = fig.add_gridspec(len(imgs), ncols, height_ratios=row_h,
                          hspace=space, wspace=space)
    for i, row in enumerate(imgs):
        span = ncols // len(row)
        for j, im in enumerate(row):
            ax = fig.add_subplot(gs[i, j * span:(j + 1) * span])
            ax.imshow(im)
            ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    return fig
