"""2-D KDE contour projections of NMR fit-quality space.

Ports the fovea ``FitResults.contour`` view: density contours of the good-fit
population across pairs of fit descriptors (FWHM, R², height, area, centre,
adjusted R²), with flagged spectra overlaid as red points. Returns a Figure;
never calls ``plt.show``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from . import style


def _clip(arr, lo=1, hi=99):
    a = np.asarray(arr, dtype=float)
    return np.clip(a, np.nanpercentile(a, lo), np.nanpercentile(a, hi))


def _kde_contour(ax, xg, yg, xb, yb, xlabel, ylabel, flag_label, n=150):
    from scipy.stats import gaussian_kde

    finite = np.isfinite(xg) & np.isfinite(yg)
    xg, yg = xg[finite], yg[finite]
    drawn = False
    if len(xg) >= 5:
        try:
            xlo, xhi = np.nanpercentile(xg, 1), np.nanpercentile(xg, 99)
            ylo, yhi = np.nanpercentile(yg, 1), np.nanpercentile(yg, 99)
            xm, ym = (xhi - xlo) * 0.1, (yhi - ylo) * 0.1
            xi = np.linspace(xlo - xm, xhi + xm, n)
            yi = np.linspace(ylo - ym, yhi + ym, n)
            xx, yy = np.meshgrid(xi, yi)
            kde = gaussian_kde(np.vstack([xg, yg]))  # raises on a degenerate dim
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            cf = ax.contourf(xx, yy, zz, levels=12, cmap="Blues")
            ax.contour(xx, yy, zz, levels=12, colors=style.MODEL, linewidths=0.4, alpha=0.6)
            ax.figure.colorbar(cf, ax=ax, label="Density")
            drawn = True
        except np.linalg.LinAlgError:
            drawn = False  # singular covariance (a flat dimension) -> scatter
    if not drawn:
        ax.scatter(xg, yg, s=3, alpha=0.3, color=style.MODEL)
    ax.scatter(xb, yb, c=style.FAIL, s=1, alpha=0.3, zorder=5, label=flag_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, markerscale=4)
    ax.grid(True, alpha=0.2)


def fit_quality_contours(df, *, quality_col="quality_pass", r2_threshold=0.99,
                         figsize=(14, 16)):
    """Grid of KDE contour projections of fit-quality space.

    ``df`` needs the columns ``r2``, ``fwhm``, ``height``, ``area_total`` and
    optionally ``r2_adj`` / ``center``. Good/flagged split comes from
    ``quality_col`` when present, else ``r2 >= r2_threshold``. Returns a Figure.
    """
    r2 = df["r2"].to_numpy()
    if quality_col in df.columns:
        good = df[quality_col].to_numpy().astype(bool)
        flag_label = "Flagged (quality map)"
    else:
        good = r2 >= r2_threshold
        flag_label = f"Flagged (R²<{r2_threshold})"
    bad = ~good
    if good.sum() == 0:
        good = np.ones(len(r2), dtype=bool)
        bad = ~good

    r2_p = _clip(r2)
    fwhm_p = _clip(df["fwhm"].to_numpy())
    h_p = _clip(df["height"].to_numpy())
    area_p = _clip(df["area_total"].to_numpy())
    r2adj_p = _clip(df["r2_adj"].to_numpy()) if "r2_adj" in df.columns else None
    center_p = _clip(df["center"].to_numpy()) if "center" in df.columns else None

    fig, axes = plt.subplots(3, 2, figsize=figsize)
    plt.subplots_adjust(hspace=0.45, wspace=0.40)

    _kde_contour(axes[0, 0], fwhm_p[good], r2_p[good], fwhm_p[bad], r2_p[bad],
                 "FWHM (ppm)", "R²", flag_label)
    axes[0, 0].set_title("Linewidth vs fit quality")
    _kde_contour(axes[0, 1], h_p[good], r2_p[good], h_p[bad], r2_p[bad],
                 "Peak height", "R²", flag_label)
    axes[0, 1].set_title("Intensity vs fit quality")
    _kde_contour(axes[1, 0], fwhm_p[good], area_p[good], fwhm_p[bad], area_p[bad],
                 "FWHM (ppm)", "Peak area", flag_label)
    axes[1, 0].set_title("Linewidth vs peak area")

    if r2adj_p is not None:
        _kde_contour(axes[1, 1], fwhm_p[good], r2adj_p[good], fwhm_p[bad], r2adj_p[bad],
                     "FWHM (ppm)", "R²_adj", flag_label)
        axes[1, 1].set_title("Linewidth vs adjusted R²")
    else:
        axes[1, 1].set_visible(False)

    if center_p is not None:
        _kde_contour(axes[2, 0], center_p[good], area_p[good], center_p[bad], area_p[bad],
                     "Chemical shift (ppm)", "Peak area", flag_label)
        axes[2, 0].set_title("Chemical shift vs peak area")
        if r2adj_p is not None:
            _kde_contour(axes[2, 1], center_p[good], r2adj_p[good], center_p[bad],
                         r2adj_p[bad], "Chemical shift (ppm)", "R²_adj", flag_label)
            axes[2, 1].set_title("Chemical shift vs adjusted R²")
        else:
            axes[2, 1].set_visible(False)
    else:
        axes[2, 0].set_visible(False)
        axes[2, 1].set_visible(False)

    fig.suptitle("Fit quality space", fontsize=13)
    return fig
