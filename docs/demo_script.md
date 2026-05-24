# Demo Script

This demo script provides a high-level workflow for running the motion estimation comparison.

1. Install dependencies from `requirements.txt`.
2. Place sample videos into `data/raw_videos/`.
3. Run the main entry point:
   ```bash
   python -m src.main
   ```
4. Check `outputs/` for generated motion vectors, residual frames, visualizations, and charts.

## Notes
- Use `notebooks/experiments.ipynb` for exploratory analysis.
- Update `src/utils/config.py` to change dataset paths or algorithm parameters.
