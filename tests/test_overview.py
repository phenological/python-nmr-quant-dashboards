"""Tests for the full-spectrum overview panel."""

import matplotlib.pyplot as plt
import numpy as np

from nmr_quant_dashboards import draw_overview


def test_rois_shaded(gauss):
    x = np.linspace(0.0, 10.0, 2000)
    y = gauss(x, 5.0, 3.0, 0.1) + gauss(x, 8.0, 7.0, 0.1)
    rois = [(2.8, 3.2, "A", True), (6.8, 7.2, "B", False)]
    fig, ax = plt.subplots()
    draw_overview(ax, x, y, rois, folder_label="folder 42")
    assert len(ax.lines) == 1          # the spectrum
    assert len(ax.patches) == 2        # one shaded span per ROI
    plt.close(fig)


def test_missing_spectrum_placeholder():
    fig, ax = plt.subplots()
    draw_overview(ax, None, None, [], folder_label="none")  # must not raise
    assert len(ax.patches) == 0
    assert len(ax.texts) >= 1          # placeholder text
    plt.close(fig)
