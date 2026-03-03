# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git
- Author: Will Cobb <will.cobb@sugarcreekresearch.com>
- Never add Co-Authored-By lines — all commits use the primary author only

## Environment
- **Never use the system Python or pyenv** — always activate a project venv
- Use `.venv-local/` when running natively (outside a container)
- Use `.venv/` when running inside a container
- Python 3.12+

## Build
```bash
# Editable install with Meson (recompiles on import)
make dev
# Or directly:
pip install --verbose --no-build-isolation --editable .

# Clean rebuild (rarely needed)
make clean
```

## Testing
```bash
# Run full discriminant analysis tests
pytest sklearn/tests/test_discriminant_analysis.py

# Run a single test
pytest sklearn/tests/test_discriminant_analysis.py::test_lda_partial_fit_svd_batch_equivalence -xvs

# Run with a keyword filter
pytest sklearn/tests/test_discriminant_analysis.py -k "partial_fit" -xvs
```
Pytest config is in `pyproject.toml` (10 min timeout, `--import-mode=importlib`).

## Linting
```bash
# Ruff (formatting + linting, 88 char line length)
ruff check sklearn/discriminant_analysis.py
ruff format sklearn/discriminant_analysis.py

# Pre-commit hooks
pre-commit run --all-files
```

## Project Context
This is a **fork of scikit-learn** adding `partial_fit` support to `LinearDiscriminantAnalysis`.

**Current status:**
- **Eigen/LSQR solvers**: `partial_fit` complete and working (covariance-based, Chan's pairwise moment-merge)
- **SVD solver**: `partial_fit` in progress — finalizing streaming SVD implementation
- **Array API**: Experimental backend acceleration support (NumPy, PyTorch, MPS)

**Key files:**
- `sklearn/discriminant_analysis.py` — Main implementation
  - `_partial_fit_svd()` — Streaming SVD with deferred attribute reconstruction
  - `_partial_fit_covariance()` — Incremental update for eigen/lsqr solvers
- `sklearn/tests/test_discriminant_analysis.py` — partial_fit tests
- `doc/whats_new/upcoming_changes/sklearn.discriminant_analysis/99999.feature.rst` — Changelog entry
- `output/jupyter-notebook/` — Benchmark notebooks

## AI Disclosure (from AGENTS.md)
Every PR description must include: "This pull request includes code written with the assistance of AI. The code has **not yet been reviewed** by a human." — update after human review occurs.
