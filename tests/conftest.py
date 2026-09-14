"""Test config: force the headless Agg backend, provide a synthetic FitPanel."""

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from nmr_quant_dashboards import FitPanel  # noqa: E402


def _gauss(x, height=10.0, center=0.0, width=0.05):
    return height * np.exp(-0.5 * ((x - center) / width) ** 2)


@pytest.fixture
def panel():
    x = np.linspace(-1.0, 1.0, 400)
    y = _gauss(x)
    return FitPanel(x=x, y=y, y_model=y.copy(), label="Acetate",
                    quality=True, metrics={"r2": 0.999, "srr": 150.0, "fwhm": 0.11})


@pytest.fixture
def gauss():
    return _gauss
