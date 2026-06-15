# 2502b — Motion Estimation Algorithm Comparison

This repository implements, visualizes, and compares two block-based motion estimation algorithms used in video compression: **Full Search** (exhaustive) and **Diamond Search** (fast heuristic). It includes preprocessing tools, block-matching implementations, residual analysis, visualization modules, and a Streamlit dashboard for interactive analysis.

---

## Repository Layout & Core Architecture

The codebase follows a highly modular structure. All computational logic is contained within the `src/` directory, while `ui/` controls the dashboard layout.

```text
├── data/                             # Project Datasets (Ignored by Git, managed at runtime)
│   ├── raw_videos/                   # Source video files (e.g., low_motion.mp4)
│   └── extracted_frames/             # Target and reference frame sequences
├── docs/                             # Documentation and performance logs
│   ├── demo_script.md                # Reference script for the demo workflow
│   └── fs_vs_diamond_search_tradeoff.md # Experimental trade-off analysis
├── notebooks/                        # Jupyter Notebooks for exploratory research
├── src/                              # Source Code Directory
│   ├── analysis/                     # Quantitative metrics reporting & data collation
│   │   ├── comparison.py             # Math comparisons (Speedup, Complexity, Accuracy loss)
│   │   └── reporting.py              # Generates markdown and CSV summary tables
│   ├── demo/                         # Demo scripting and video slicing logic
│   │   ├── frame_selector.py         # Dynamic frame picker based on motion intensity/uniformity
│   │   ├── video_clips.py            # Extracts short clips from source videos
│   │   └── run_demo.py               # Main pipeline execution for demo workflows
│   ├── motion_estimation/            # Core Block-Matching Algorithms
│   │   ├── block_matching.py         # SAD, MAD, MSE objective cost evaluators
│   │   ├── full_search.py            # Exhaustive baseline motion estimator
│   │   ├── diamond_search.py         # LDSP/SDSP pattern fast heuristic search
│   │   └── motion_vector.py          # Data structures for handling vector arrays
│   ├── preprocessing/                # Video reading and image formatting
│   │   ├── frame_extractor.py        # Frame-by-frame extraction via OpenCV
│   │   └── grayscale_converter.py    # Color space optimization for block matching
│   ├── residual/                     # Performance evaluations
│   │   ├── residual_energy.py        # Metrics for total residual cost calculation
│   │   └── residual_generator.py     # Generates int16 signed errors and uint8 mid-gray offsets
│   ├── utils/                        # System configurations and IO helper files
│   │   ├── config.py                 # Hyperparameters (Block size, search window, directory paths)
│   │   ├── file_handler.py           # Multi-format logging and export handlers
│   │   └── metrics.py                # Mathematical helpers for MSE and PSNR
│   ├── visualization/                # Plotting components
│   │   ├── comparison_chart.py       # Comparative runtime and MSE data graphing
│   │   ├── vector_field.py           # Matplotlib Quiver mapping for motion vectors
│   │   └── visualization_manager.py  # Central manager handling output exports
│   ├── finalize.py                   # Automatic project packaging pipeline
│   ├── main.py                       # CLI Entrypoint for full benchmark experiments
│   └── workflow.py                   # Higher-level pipeline orchestration workflow
├── ui/                               # Streamlit Web Interfaces
│   ├── components.py                 # Custom UI layouts, sidebar states, and parameters
│   ├── dashboard.py                  # Streamlit central app structure
│   ├── helpers.py                    # Metadata parser and caching utilities
│   └── visualizations.py             # Rendering layer for dashboard vector fields
├── app.py                            # Streamlit entrypoint script
├── requirements.txt                  # Python runtime dependencies
└── README.md                         # This file

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

Core Algorithm Implementation Details
Mathematical Cost Functions
Block matching defaults to SAD (Sum of Absolute Differences) for optimal hardware emulation efficiency, with multi-metric evaluators (MAD, MSE) available inside src/motion_estimation/block_matching.py.

Residual Frame Normalization
Residuals are calculated using signed data types to prevent overflow/underflow truncation:

         Signed Error = Target Frame(int16) - Predicted Frame(int16)

For intuitive visualization, residual gray-scale outputs map Zero Error to mid-gray (128), positive deviations to bright tones (>128), and negative errors to dark tones (<128).
Contributing
------------

- Please open issues or PRs for improvements.
- Run tests and ensure new code includes unit tests where appropriate.

Authors
-------

- Triệu Tiến Nguyên - 202414651
- Nguyễn Lâm Tuấn Linh - 202414637

License
-------

This project is provided under the terms of the repository LICENSE file.

