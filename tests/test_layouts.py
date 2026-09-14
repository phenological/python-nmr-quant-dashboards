"""Tests for the multi-panel dashboard layouts."""

import copy

import matplotlib.pyplot as plt
import numpy as np

from nmr_quant_dashboards import dashboard_compact, dashboard_full


def _panels(panel, k):
    return [copy.copy(panel) for _ in range(k)]


def test_dashboard_full_with_overview(panel):
    x = np.linspace(0.0, 10.0, 500)
    y = np.zeros_like(x)
    overview = (x, y, [(2.8, 3.2, "A", True)], "folder 1")
    fig = dashboard_full(_panels(panel, 2), overview=overview, title="QA")
    assert fig is not None
    # 1 overview + 2 panels x (fit + residual) = 5 axes
    assert len(fig.axes) == 5
    plt.close(fig)


def test_dashboard_full_without_overview(panel):
    fig = dashboard_full(_panels(panel, 3))
    assert len(fig.axes) == 6  # 3 x (fit + residual)
    plt.close(fig)


def test_dashboard_compact_grid(panel):
    fig = dashboard_compact(_panels(panel, 3), ncols=2)
    assert len(fig.axes) == 3
    plt.close(fig)


def test_empty_panels_return_none():
    assert dashboard_full([]) is None
    assert dashboard_compact([]) is None
