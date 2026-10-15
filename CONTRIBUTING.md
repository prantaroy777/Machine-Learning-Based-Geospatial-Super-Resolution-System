# Contributing

Contributions are welcome: documentation, reproducible bug reports, tests,
performance improvements and properly validated geospatial workflows all help.
You do not need repository write access to propose a change.

## Submit a contribution

1. Read the README and model card. For a substantial design change, open a feature
   request first so the scope and validation approach can be discussed.
2. Fork the repository on GitHub, then clone your fork to your computer.
3. Create a branch: `git switch -c describe-your-change`.
4. Create a virtual environment and install the dependencies as described below.
5. Implement a focused change and add relevant regression tests.
6. Run the test suite, inspect `git diff` and `git status`, and commit the intended files.
7. Push your branch to your fork and open a pull request against the main repository.
8. Fill in the pull request template and respond to review feedback. Maintainers
   review and merge changes; opening a pull request does not grant write access.

Small documentation corrections can be submitted directly as pull requests.
By submitting a contribution, you agree that it can be distributed under this
repository's MIT license. Follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Development environment

Use Python 3.10 or 3.11 and a project virtual environment. Install `requirements.txt`
for the full suite or `requirements-core.txt` for interpolation and raster evaluation.

Run `python -m unittest discover -s tests -v` before submitting changes. On Windows,
the Streamlit test runner needs permission to use local sockets. Tests use temporary
synthetic fixtures and do not require downloaded satellite data or a GPU.

For bug reports, include the command, traceback, OS, Python/package versions,
CRS, band count, raster dimensions and pixel spacing. Share a small synthetic
reproducer when possible. Do not upload private imagery, credentials or personal paths.

Changes to interpolation, normalization, NoData handling or alignment must include
numerical regression tests. Never promote synthetic improvements as measured
satellite resolution gains. Report the interpolation baseline, data split and
sensor/degradation assumptions with any model benchmark.

The application is for a trusted local user. It accepts local filesystem paths and
is not designed as a public multi-user service. Authentication, isolation, quotas,
job scheduling and an upload-retention policy are required before hosting it publicly.
