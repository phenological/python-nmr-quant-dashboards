# nmr-quant-dashboards

Matplotlib figures for inspecting NMR peak fits and their quality: per-signal
fit panels (data, model, optional components and baseline), a residual strip,
and a full-spectrum overview with fitted ROIs shaded by pass/fail.

It is the visualisation layer that `nmr-quant` and `python-nmr-spectra-processing`
deliberately leave out. It depends on **only numpy and matplotlib**: panels take
pre-computed arrays (`FitPanel`), so the caller evaluates the model (for
example via `nmr_quant.QuantResults.model_curve`) and this package only draws.
Functions return `Figure`/`Axes` and never touch disk.

## Public API

- `FitPanel` — plain-array input for one signal's fit (`x`, `y`, `y_model`,
  optional `components`, `baseline`, `metrics`, `quality`).
- `draw_fit_panel(ax, panel, ...)` — data + model (+ components/baseline).
- `draw_residual(ax, panel, ...)` — residual with a zero line.
- `draw_overview(ax, full_x, full_y, rois, ...)` — full spectrum, ROIs shaded.
- `dashboard_full(panels, overview=..., title=...)` — overview + per-signal
  fit/residual figure.
- `dashboard_compact(panels, ...)` — a compact grid of fit panels.

## Install (dev)

```bash
uv sync
uv run pytest
```

## License

MIT.
