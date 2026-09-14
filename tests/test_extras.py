"""Tests for badges, median overlay, richer residual, and contour grid."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nmr_quant_dashboards import (
    FitPanel,
    draw_badges,
    draw_median,
    draw_residual,
    fit_quality_contours,
)

_METRICS = {"kde_score": 0.8, "fwhm": 0.011, "area_total": 1.2, "center": 0.001,
            "r2": 0.999, "r2_adj": 0.99, "srr": 150.0, "height": 10.0}


def test_draw_badges_counts_and_failures():
    fig, ax = plt.subplots()
    draw_badges(ax, _METRICS, failed=["r2", "area"])
    texts = [t.get_text() for t in ax.texts]
    assert len(texts) == 7  # one per present metric
    assert sum("FAIL" in t for t in texts) == 2  # r2 and area failed
    plt.close(fig)


def test_draw_badges_no_quality_uses_threshold():
    fig, ax = plt.subplots()
    draw_badges(ax, _METRICS, failed=None)  # no quality map
    texts = [t.get_text() for t in ax.texts]
    assert all("PASS" in t for t in texts)  # r2=0.999 >= 0.99, rest default pass
    plt.close(fig)


def test_draw_median_two_lines():
    x = np.linspace(-1, 1, 200)
    y = np.exp(-0.5 * (x / 0.05) ** 2)
    med = 0.9 * y
    fig, ax = plt.subplots()
    draw_median(ax, x, y, med)
    assert len(ax.lines) == 2
    plt.close(fig)


def test_residual_percent_and_area():
    x = np.linspace(-1, 1, 400)
    y = 10 * np.exp(-0.5 * (x / 0.05) ** 2)
    panel = FitPanel(x=x, y=y, y_model=0.95 * y,
                     metrics={"height": 10.0, "center": 0.0})
    fig, ax = plt.subplots()
    draw_residual(ax, panel, as_percent=True, peak_height=10.0, show_area=True)
    assert len(ax.lines) == 2                 # zero line + residual
    assert "%" in ax.get_ylabel()
    assert len(ax.collections) >= 1           # signed-area shading
    plt.close(fig)


def test_fit_quality_contours_returns_figure():
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame({
        "r2": np.r_[0.995 + rng.standard_normal(n) * 1e-3, [0.5, 0.7]],
        "fwhm": np.r_[0.011 + rng.standard_normal(n) * 5e-4, [0.05, 0.001]],
        "height": np.r_[10 + rng.standard_normal(n), [50, 1]],
        "area_total": np.r_[1.0 + rng.standard_normal(n) * 0.05, [9, 0.01]],
        "r2_adj": np.r_[0.98 + rng.standard_normal(n) * 2e-3, [0.4, 0.6]],
        "center": np.r_[rng.standard_normal(n) * 5e-4, [0.5, -0.5]],
        "quality_pass": np.r_[np.ones(n, bool), [False, False]],
    })
    fig = fit_quality_contours(df, figsize=(10, 12))
    assert fig is not None
    assert len(fig.axes) >= 6  # 3x2 grid (+ colorbars)
    plt.close(fig)
