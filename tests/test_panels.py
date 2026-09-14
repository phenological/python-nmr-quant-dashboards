"""Tests for the panel builders (assert on artist structure, not pixels)."""

import matplotlib.pyplot as plt
import numpy as np

from nmr_quant_dashboards import FitPanel, draw_fit_panel, draw_residual, style


def test_draw_fit_panel_data_and_model(panel):
    fig, ax = plt.subplots()
    draw_fit_panel(ax, panel)
    assert len(ax.lines) == 2  # data + model
    plt.close(fig)


def test_components_and_baseline_add_lines(panel, gauss):
    x = panel.x
    panel.components = [("sat-", gauss(x, 1.0, -0.3)), ("sat+", gauss(x, 1.0, 0.3))]
    panel.baseline = np.full_like(x, 0.5)
    fig, ax = plt.subplots()
    draw_fit_panel(ax, panel)
    assert len(ax.lines) == 5  # data + model + 2 components + baseline
    plt.close(fig)


def test_show_components_false_drops_them(panel, gauss):
    panel.components = [("sat", gauss(panel.x, 1.0, 0.3))]
    fig, ax = plt.subplots()
    draw_fit_panel(ax, panel, show_components=False)
    assert len(ax.lines) == 2  # data + model only
    plt.close(fig)


def test_quality_colours_title(panel):
    panel.quality = False
    fig, ax = plt.subplots()
    draw_fit_panel(ax, panel)
    assert ax.title.get_color() == style.FAIL
    plt.close(fig)


def test_xlim_respected(panel):
    fig, ax = plt.subplots()
    draw_fit_panel(ax, panel, xlim=(0.5, -0.5))  # inverted ppm convention
    assert ax.get_xlim() == (0.5, -0.5)
    plt.close(fig)


def test_error_path_does_not_raise():
    fig, ax = plt.subplots()
    bad = FitPanel(x=np.array([]), y=np.array([]))
    draw_fit_panel(ax, bad)  # must not raise
    assert ax.get_title().startswith("ERROR")
    plt.close(fig)


def test_residual_is_data_minus_model(panel, gauss):
    panel.y_model = gauss(panel.x) * 0.9  # 10% off
    fig, ax = plt.subplots()
    draw_residual(ax, panel)
    # one zero line + one residual line
    assert len(ax.lines) == 2
    resid_line = ax.lines[1]
    assert np.allclose(resid_line.get_ydata(), panel.y - panel.y_model)
    plt.close(fig)


def test_residual_without_model(panel):
    panel.y_model = None
    fig, ax = plt.subplots()
    draw_residual(ax, panel)
    assert len(ax.lines) == 0  # placeholder text only
    plt.close(fig)
