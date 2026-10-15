# Changelog

## 0.2.1 — 2026-10-15

- Added validation for empty, missing, escaping and malformed prepared patches.
- Enforced checkpoint band/radiometry compatibility during evaluation.
- Handled deleted GUI result files and dependency import failures gracefully.
- Added command-line workflow tests and a temporary HTTP server smoke check.
- Verified fresh core-only installation and expanded CI to Python 3.10/3.11.
- Added contribution forms, pull request guidance, community/security policies,
  dependency-update configuration, and a reproducible source-release packager.
- Updated the project title and included a reproducible synthetic illustration.

## 0.2.0 — 2026-10-14

- Added training from aligned observed low/high-resolution pairs, synchronized
  augmentation, strict grid/band/radiometry checks, and early stopping.
- Added windowed reference-raster comparison with per-band and overall MAE, RMSE,
  bias, PSNR, valid coverage, spectral angle, and optional radiometric scaling.
- Matched exported bicubic interpolation to the training baseline's Keys kernel.
- Added output provenance, checkpoint fingerprints, exclusive output locks,
  failure cleanup tests, optional GIS overviews and tile progress callbacks.
- Added preview band selection, downloadable processing reports, in-app reference
  evaluation and a synthetic demo that works in a fresh checkout.
- Replaced the email-prompt launcher with a readiness-checked local launcher.
- Added an MIT license, contributor guide, model card, constrained dependencies,
  a lighter installation option, and Windows/Linux CI configuration.
- Removed empty notebook placeholders; runnable examples remain in `examples/`.

Validation: 25 local tests passed on Windows, including the Streamlit interface,
paired training, geospatial metadata/masks, numerical regressions and error paths.
GPU checkpoint inference was verified. Command-line reference comparison
and GeoTIFF export with overviews were exercised with synthetic fixtures.
GitHub Actions has not yet run. No real-satellite accuracy benchmark was performed.
