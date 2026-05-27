"""Small helper to generate a synthetic video and run the motion estimation workflow end-to-end for testing."""
from pathlib import Path
import cv2
import numpy as np
import time

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from src.workflow import run_motion_estimation_workflow


def make_synthetic_video(path: Path, width=128, height=128, frames=20, fps=10):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # moving square
        x = int((width - 32) * i / max(1, frames - 1))
        cv2.rectangle(img, (x, 32), (x + 32, 64), (255, 255, 255), -1)
        out.write(img)
    out.release()


if __name__ == '__main__':
    tmp = Path("data") / "demo_synthetic.mp4"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    print("Generating synthetic video...", tmp)
    make_synthetic_video(tmp)
    print("Running workflow...")
    t0 = time.perf_counter()
    res = run_motion_estimation_workflow(tmp, block_size=16, search_range=7, output_root=Path("outputs") / "demo_run")
    elapsed = time.perf_counter() - t0
    print(f"Workflow finished in {elapsed:.2f}s")
    print(res)
