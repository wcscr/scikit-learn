# Feedback Response: `LinearDiscriminantAnalysis.partial_fit`

Branch: `feature/lda-partial-fit`

## 1. [P0] `partial_fit` breaks when called after `fit`

**Status:** Fixed

**Changes:** In `sklearn/discriminant_analysis.py`, the accumulator initialization block was
changed from `if first_call:` to `if first_call or not hasattr(self, "_class_counts"):`. This
ensures that when `partial_fit` is called after `fit` (which sets `classes_` but not
`_class_counts`/`_unscaled_covariance`), the accumulators are re-initialized rather than
crashing on missing attributes.

**Test added:** `test_lda_partial_fit_after_fit` — calls `fit()` then `partial_fit()` and verifies
the result matches a fresh `partial_fit`-only estimator.

---

## 2. [P0] Unknown labels in `y` are silently ignored

**Status:** Fixed

**Changes:** Added validation after `validate_data` and before the update loop that checks for
labels not in `self.classes_`:

```python
unexpected = np.setdiff1d(y, self.classes_)
if len(unexpected) > 0:
    raise ValueError(
        "The target label(s) %s in y do not exist in the "
        "initial classes %s" % (unexpected, self.classes_)
    )
```

**Test added:** `test_lda_partial_fit_raises_unknown_labels` — verifies that a `ValueError`
is raised when `y` contains labels not seen during the first `partial_fit` call.

---

## 3. [P1] `priors` parameter contract differs between `fit` and `partial_fit`

**Status:** Fixed

**Changes:** The priors derivation section now mirrors `fit()` behavior. When `self.priors` is
not None, the user-supplied priors are used (with the same negative-value check and
renormalization warning). When `self.priors` is None, count-based priors are computed as before.

**Test added:** `test_lda_partial_fit_honors_priors` — verifies that explicit priors are
used for `priors_` and that the resulting `intercept_` differs from the count-based case.

---

## 4. [P1] `within_df < n_features` gate is too strict for `lsqr`

**Status:** Fixed

**Changes:** The early-return condition was changed from:
```python
if n_classes_seen < 2 or within_df < n_features:
```
to:
```python
if n_classes_seen < 2 or (self.solver == "eigen" and within_df < n_features):
```

The `lsqr` solver uses `linalg.lstsq` which handles rank-deficient matrices, so the
within-df check is only needed for the `eigen` solver.

**Test added:** `test_lda_partial_fit_lsqr_low_samples` — verifies that `lsqr` works when
`N_total - n_classes < n_features` (4 samples, 2 classes, 10 features).

---

## 5. [P1] Early-return path can leave a half-fitted estimator

**Status:** Fixed

**Changes:** Added a `__sklearn_is_fitted__` method to `LinearDiscriminantAnalysis` that
returns `hasattr(self, "coef_")`. This is the standard scikit-learn pattern (used by
`Pipeline`, `FeatureUnion`, `FunctionTransformer`) — `check_is_fitted` checks for this
method first. Since `coef_` is only set after the solver runs, early-return paths that set
accumulator attributes (`means_`, `priors_`) will not cause the estimator to appear fitted.

**Test added:** `test_lda_partial_fit_early_return_not_fitted` — verifies that `predict`
raises `NotFittedError` (not `AttributeError`) after an early-return `partial_fit` where
only one class has been observed.

---

## 6. [P2] Array API support inconsistent

**Status:** No change needed

The `partial_fit` method explicitly blocks `solver='svd'` (the only solver with array API
support) and raises `NotImplementedError`. The supported solvers (`eigen`, `lsqr`) are
already NumPy-only in the batch `fit` path as well, so users cannot reach the NumPy-only
code path with array API inputs.

---

## 7. [P3] Contribution-guideline compliance

**Status:** Fixed

**7a. Version tag:** Changed `.. versionadded:: 1.7` to `.. versionadded:: 1.9` to match
the current dev cycle (`1.9.dev0`).

**7b. Changelog:** Created
`doc/whats_new/upcoming_changes/sklearn.discriminant_analysis/99999.feature.rst`
with a changelog entry for the new `partial_fit` feature.

**7c. Long lines in tests:** Fixed two lines exceeding 88 characters:
- Line 904: broke the `partial_fit` call across multiple lines
- Line 976: broke the `pytest.raises` call across multiple lines

---

## Verification

- All 79 tests pass: `python -m pytest sklearn/tests/test_discriminant_analysis.py -v`
- 4 new tests added covering all code fixes
