# 2502b — Motion Estimation Algorithm Comparison

## Project Overview
This university multimedia compression project compares Full Search (FS) and Diamond Search motion estimation algorithms for video compression. The repository is structured to support preprocessing, motion estimation, residual generation, visualization, and analysis.

## Objectives
- Compare Full Search and Diamond Search motion estimation performance
- Generate motion vectors and residual frames
- Analyze residual energy for compression efficiency
- Visualize motion vector fields and algorithm comparisons
- Measure runtime differences and chart results

## Folder Structure
```
2502b-motion-estimation-comparison/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw_videos/
│   │   ├── low_motion.mp4
│   │   ├── medium_motion.mp4
│   │   └── high_motion.mp4
│   │
│   ├── extracted_frames/
│   │   ├── low_motion/
│   │   ├── medium_motion/
│   │   └── high_motion/
│   │
│   └── processed/
│
├── src/
│   ├── main.py
│   │
│   ├── preprocessing/
│   │   ├── frame_extractor.py
│   │   └── grayscale_converter.py
│   │
│   ├── motion_estimation/
│   │   ├── full_search.py
│   │   ├── diamond_search.py
│   │   ├── block_matching.py
│   │   └── motion_vector.py
│   │
│   ├── residual/
│   │   ├── residual_generator.py
│   │   └── residual_energy.py
│   │
│   ├── visualization/
│   │   ├── vector_field.py
│   │   ├── residual_plot.py
│   │   └── comparison_chart.py
│   │
│   └── utils/
│       ├── config.py
│       ├── metrics.py
│       └── file_handler.py
│
├── outputs/
│   ├── motion_vectors/
│   │   ├── fs/
│   │   └── diamond/
│   │
│   ├── residual_frames/
│   │
│   ├── vector_visualizations/
│   │
│   ├── charts/
│   │
│   └── logs/
│
├── notebooks/
│   └── experiments.ipynb
│
├── docs/
│   ├── report/
│   ├── screenshots/
│   └── demo_script.md
│
└── tests/
    ├── test_full_search.py
    ├── test_diamond_search.py
    └── test_residual.py
```

## Installation
1. Create and activate a Python 3.11+ virtual environment.
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Run
Run the project from the repository root using the package entry point:
```bash
python -m src.main
```

Run the demo workflow to generate short clips, representative frames, and comparison screenshots:
```bash
python -m src.demo.run_demo
```

The demo workflow writes outputs to:
- `outputs/demo_videos/`
- `outputs/demo_frames/`
- `outputs/demo_screenshots/`

Run the main comparison workflow to generate algorithm comparison tables:
```bash
python -m src.main
```

The main comparison workflow writes experiment reports to:
- `outputs/reports/`

Run the final packaging workflow to prepare the final project folder:
```bash
python -m src.finalize
```

The final project workflow writes:
- `outputs/final_project/`

## Expected Outputs
- Motion vector files for FS and Diamond Search
- Residual frame images
- Residual energy comparison metrics
- Motion vector field visualizations
- Runtime and algorithm comparison charts

## Algorithm Overview
- **Full Search**: exhaustive block matching over a search window.
- **Diamond Search**: fast search heuristic that reduces candidate evaluations.
- **Block Matching**: similarity metrics such as SAD, MAD, and MSE are used to find the best motion matches.

## Future Improvements
- Add real codec integration for bitstream generation
- Support additional fast search algorithms
- Add quantitative rate-distortion analysis
- Improve visualizations and reporting

## Team Members
- Triệu Tiến Nguyên
- Nguyễn Lâm Tuấn Linh

## Quickstart

- Create and activate the Python virtual environment (Python 3.10+ recommended):

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

- Run tests:

```bash
python -m pytest -q
```

## Recent Changes

- Added optional `save_path` for motion vector exporters in `src/motion_estimation`.
- Fixed a signature bug in `diamond_search` and improved metric handling.
- Micro-optimized block-matching metrics and improved vector visualizations.

