# 2502b — Motion Estimation Algorithm Comparison

This repository implements and compares two block-based motion estimation algorithms used in video compression: Full Search (exhaustive) and Diamond Search (fast heuristic). It includes preprocessing tools, motion estimation implementations, residual generation and analysis, visualizations, and a Streamlit dashboard for interactive demos.

**Key goals:**
- Compare estimation quality (residual energy) and runtime between Full Search and Diamond Search
- Produce motion vector CSVs, reconstructed/predicted frames, and signed residual images
- Generate charts and vector-field visualizations for analysis and reports

**Repository layout (important paths):**

- `data/raw_videos/` — source video files used for experiments
- `data/extracted_frames/` — extracted frames (per-video folders)
- `src/` — source code; key modules:
    - `src/main.py` — main CLI entry for running experiments
    - `src/workflow.py` — high-level workflow orchestration
    - `src/demo/run_demo.py` — small demo runner for quick experiments
    - `src/motion_estimation/` — algorithms: `full_search.py`, `diamond_search.py`
    - `src/residual/` — `residual_generator.py`, `residual_energy.py`
    - `src/visualization/` — plotting helpers and vector field rendering
- `outputs/` — generated artifacts (ignored by default); subfolders introduced at runtime:
    - `outputs/motion_vectors/{fs,diamond}/` — CSV motion vectors
    - `outputs/residual_frames/` — signed residual images (uint8 mapped with 128=center)
    - `outputs/reports/` — CSV/MD experiment reports and runtime tables
    - `outputs/visualizations/` — charts and comparison images

Prerequisites
-------------
- Python 3.10+ (3.11 recommended)
- A working virtual environment (venv, conda, etc.)
- System packages: OpenCV (cv2), NumPy, Matplotlib, Streamlit (optional)

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell
pip install -r requirements.txt
```

Running experiments
-------------------

1) Quick demo (small, fast)

```powershell
python -m src.demo.run_demo
```

This writes demo outputs into `outputs/demo_*` and `outputs/visualizations/`.

2) Full comparison workflow

```powershell
python -m src.main
```

This runs the configured experiments (see `src/utils/config.py`) and writes motion vectors, residual frames, and reports to `outputs/`.

3) Run a single workflow step (example: extract frames)

```powershell
python -m src.preprocessing.frame_extractor --input data/raw_videos/high_motion.mp4 --out data/extracted_frames/high_motion
```

Testing
-------

Run the test suite with:

```powershell
python -m pytest -q
```

Streamlit dashboard
-------------------

Start the interactive demo (optional):

```powershell
streamlit run app.py
```

The dashboard can be used to run short experiments, visualize motion vectors, and download generated outputs.

Notes about residuals and outputs
--------------------------------
- Residuals are computed as `signed = target.astype(int16) - predicted` and are saved as uint8 images with zero mapped to 128 (mid-gray). This preserves sign information when visualizing residuals.
- `outputs/` is in `.gitignore` by default to avoid committing large artifacts. Only commit metadata and code. If you need to publish specific results, export the relevant CSV/PNG and commit them separately under a dedicated `results/` folder.

Contributing
------------

- Please open issues or PRs for improvements.
- Run tests and ensure new code includes unit tests where appropriate.

Authors
-------

- Triệu Tiến Nguyên
- Nguyễn Lâm Tuấn Linh

License
-------

This project is provided under the terms of the repository LICENSE file.

