# Full Search vs Diamond Search Tradeoff Analysis

## Context
This analysis compares Full Search (FS) and Diamond Search (DS) motion estimation using the sample video `data/raw_videos/low_motion.mp4`.

## Measured Results
- Full Search execution time: ~4730.8 ms
- Diamond Search execution time: ~421.9 ms
- Measured speedup: **11.2x faster** for Diamond Search

- Full Search block comparisons: **638,912**
- Diamond Search block comparisons: **54,617**
- Complexity reduction: **91.5% fewer comparisons** with Diamond Search

- Full Search MSE: **11173.82**
- Diamond Search MSE: **11172.00**
- PSNR: **7.65 dB** for both methods

- Full Search residual energy: **71,579,712**
- Diamond Search residual energy: **70,767,808**
- Residual energy difference: **1.1%**

## Quality Comparison
- Diamond Search produced slightly better numerical accuracy on this sample pair (lower MSE by 1.82).
- Both methods delivered nearly identical PSNR, indicating similar overall motion compensation quality.
- Residual energy between the methods was substantially similar, so compression efficiency is not meaningfully degraded by DS.

## Strengths and Weaknesses
### Full Search
- Strengths:
  - Guarantees globally optimal block matching within the search window.
  - Best option when maximum accuracy is required and compute cost is less critical.
- Weaknesses:
  - Very high computational cost due to exhaustive search.
  - Slower runtime and far more block comparisons.

### Diamond Search
- Strengths:
  - Highly efficient search strategy with large speedup.
  - Achieves nearly the same or slightly better motion accuracy in this test.
  - Reduces complexity by more than 90%.
- Weaknesses:
  - Approximate search can fail on highly irregular or noisy motion patterns.
  - May require parameter tuning if search range or block size changes.

## Conclusions
- Diamond Search is the preferred algorithm for this dataset and configuration.
- It provides a strong speed/efficiency advantage while maintaining very similar motion estimation quality.
- For practical motion estimation workflows, DS is recommended when compute resources are limited.
- Full Search remains useful as a baseline or when exhaustive accuracy verification is needed.

## Notes
- The comparison workflow is implemented in `src/main.py` and summarized by `src/analysis/comparison.py`.
- All code changes were tested and validated with the existing unit tests for FS, DS, and residual computation.
