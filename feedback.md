# Review Feedback: `LinearDiscriminantAnalysis.partial_fit`

Date: 2026-02-27
Branch: `feature/lda-partial-fit`
Commit reviewed: `d2b169a754`

## Findings

1. **[P0] `partial_fit` breaks when called after `fit`**
   - References: `sklearn/discriminant_analysis.py:802`, `sklearn/discriminant_analysis.py:816`, `sklearn/discriminant_analysis.py:837`
   - Problem: `first_call` becomes `False` after `fit` because `classes_` exists, but `_class_counts` and `_unscaled_covariance` are initialized only on the first `partial_fit` call. Calling `partial_fit` after `fit` can dereference missing state.

2. **[P0] Unknown labels in `y` are silently ignored**
   - Reference: `sklearn/discriminant_analysis.py:824`
   - Problem: The update loop iterates only over `self.classes_`. Labels not in `self.classes_` are ignored instead of raising a `ValueError`, which can silently drop training samples.

3. **[P1] `priors` parameter contract differs between `fit` and `partial_fit`**
   - References: `sklearn/discriminant_analysis.py:699`, `sklearn/discriminant_analysis.py:849`
   - Problem: `fit` honors user-provided `priors`; `partial_fit` always recomputes `priors_` from observed counts, changing estimator behavior depending on training path.

4. **[P1] `within_df < n_features` gate is too strict for `lsqr`**
   - References: `sklearn/discriminant_analysis.py:856`, `sklearn/discriminant_analysis.py:504`
   - Problem: `lsqr` uses `linalg.lstsq` and can handle singular covariance. The early return can block valid `lsqr` fits in high-dimensional or low-sample regimes.

5. **[P1] Early-return path can leave a half-fitted estimator**
   - References: `sklearn/discriminant_analysis.py:817`, `sklearn/discriminant_analysis.py:857`, `sklearn/linear_model/_base.py:361`, `sklearn/linear_model/_base.py:365`
   - Problem: `means_` and `priors_` are set before returning, but `coef_` may be missing. Downstream prediction calls can fail with `AttributeError`.

6. **[P2] Array API support is inconsistent in `partial_fit`**
   - References: `sklearn/discriminant_analysis.py:1015`, `sklearn/discriminant_analysis.py:809`, `sklearn/discriminant_analysis.py:830`
   - Problem: Class tag advertises `array_api_support=True`, but `partial_fit` is implemented with NumPy-only operations (`np.mean`, `np.outer`, fixed NumPy dtypes) instead of namespace-aware code.

7. **[P3] Contribution-guideline compliance gaps**
   - References: `sklearn/discriminant_analysis.py:771`, `sklearn/__init__.py:45`, `doc/developers/contributing.rst:461`, `pyproject.toml:127`, `sklearn/tests/test_discriminant_analysis.py:904`, `sklearn/tests/test_discriminant_analysis.py:976`
   - Problems:
     - `.. versionadded:: 1.7` does not match the current dev cycle (`1.9.dev0`).
     - No changelog fragment for a user-facing feature was added.
     - New tests contain lines beyond the 88-character style limit.

## Validation Notes

- `pytest` could not be run in the current environment (`No module named pytest`).
- Direct runtime probes also failed due missing `numpy` in the active interpreter.

This pull request includes code written with the assistance of AI.
The code has **not yet been reviewed** by a human.
