# Experiments Notebook Notes

This file summarizes how to run the experiments and notes about recent code changes.

## How to run

1. Activate virtualenv:

```
.\.venv\Scripts\activate
```

2. Convert frames / prepare data as described in the repo README.

3. Run the notebook `notebooks/experiments.ipynb` to reproduce visualizations.

## Recent Code Notes

- Motion estimation functions (`full_search` and `diamond_search`) now accept an optional `save_path` argument to export motion vectors as CSV (`x,y,dx,dy`).
- Residual generation returns signed residuals (`target - predicted`) as `int16` arrays.
- Block-matching metrics (SAD/MAD/MSE) were optimized for performance.
- Vector visualizations now use `quiver` for improved clarity and speed.

## Next steps for the notebook

- Update `notebooks/experiments.ipynb` to call the updated motion estimation functions with `save_path` where appropriate.
- Add a cell to read exported CSV motion vectors and overlay them on frames using `src.visualization.vector_field.overlay_vectors_on_frame`.
