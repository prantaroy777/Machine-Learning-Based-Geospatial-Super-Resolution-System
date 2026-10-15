# Release verification — 2026-10-15

## Completed locally

| Check | Result |
| --- | --- |
| Full Python 3.10 Windows environment | 29 tests passed |
| Fresh installation from requirements-core.txt | Installed successfully; no dependency conflicts |
| Fresh environment without PyTorch | 22 tests passed; 7 model-only tests skipped as expected |
| GUI interaction | Demo, file selection, processing, reference evaluation, persistent results, and deleted-output handling tested |
| CLI workflow | Generated synthetic data, exported 10 m GeoTIFF, inspected grid/extent, compared reference, and checked overwrite rejection |
| Real-pair code path | Synthetic aligned LR/HR fixtures exercised preparation, synchronized augmentation and training |
| GPU | Small model forward/backward pass completed with finite output |
| Server | Health endpoint and HTML returned successfully in the fresh environment; temporary server stopped afterward |
| Source | Python compilation, configuration YAML parsing and source-release integrity checked |

Tests also cover NoData, tiling equivalence, radiometric scaling, interpolation
kernel consistency, invalid datasets/checkpoints, geographic separation, output
locks and partial-file cleanup. `python scripts/check_server.py` runs the HTTP
startup check; `python -m unittest discover -s tests -v` runs the regression suite.

## Not established by these checks

- Real-satellite reconstruction accuracy or effective 10 m resolving power.
- Every sensor, raster size, GPU, operating system or dependency combination.
- Public multi-user deployment security, availability or a formal security audit.
- Successful GitHub Actions execution before the project is uploaded.

CI is configured to test Windows/Linux with Python 3.10/3.11 and a separate
core-only installation. Check actual CI results after publishing. Dependency
ranges allow newer releases, so future installations can differ from this run.
The release is a tested research prototype, not a promise of defect-free software.
