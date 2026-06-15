from pathlib import Path
from src.utils.config import Config
from src.workflow import run_motion_estimation_workflow

videos = [
    Config.RAW_VIDEOS_DIR / "low_motion.mp4",
    Config.RAW_VIDEOS_DIR / "high_motion.mp4",
    Config.RAW_VIDEOS_DIR / "noisy_motion.mp4",
]

for v in videos:
    if v.exists():
        print(f"Running workflow for {v.name}")
        try:
            out = run_motion_estimation_workflow(v)
            print(f"OK: {v.name} -> residual energy: {out['residual_stats']['energy']}")
        except Exception as e:
            print(f"ERROR running {v.name}: {e}")
    else:
        print(f"Skipped missing video: {v.name}")
