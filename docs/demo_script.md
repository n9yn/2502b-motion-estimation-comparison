# Demo Script

This demo script provides a workflow to generate video clips, select representative frames, and create comparison screenshots.

## Steps

1. Install dependencies from `requirements.txt`.
2. Place sample videos into `data/raw_videos/`.
3. Run the demo workflow:
   ```bash
   python -m src.demo.run_demo
   ```
4. Verify generated outputs:
   - `outputs/demo_videos/` for auto-generated demo clips
   - `outputs/demo_frames/` for representative selected frames
   - `outputs/demo_screenshots/` for motion estimation comparison screenshots
5. Run the main comparison workflow to generate performance tables:
   ```bash
   python -m src.main
   ```
6. Verify generated experiment tables in `outputs/reports/`.
7. Prepare the final project folder for delivery:
   ```bash
   python -m src.finalize
   ```
8. Verify `outputs/final_project/` contains copied output sections.
   ```bash
   python -m src.main
   ```

## Task Checklist

- [x] Create demo video clips
- [x] Select representative frames
- [x] Prepare comparison screenshots
- [x] Test playback by opening generated MP4 clips in a media player

## Notes
- Use `notebooks/experiments.ipynb` for exploratory analysis.
- Update `src/utils/config.py` to change dataset paths or algorithm parameters.
