# Implementation Plan: Address LDA partial_fit Review Comments

Based on `review_lda_partial_fit.md`, here is the step-by-step plan organized by priority.

---

## Phase 1: Must-Fix Issues

### Step 1: Fix changelog attribution (Issue #3)
- **File:** `doc/whats_new/upcoming_changes/sklearn.discriminant_analysis/99999.feature.rst`
- **Change:** Replace `:user:\`Will Cobb <willcobb>\`` with `:user:\`Will Cobb <wcscr>\`` to match the actual GitHub username.

### Step 2: Fix error type inconsistency (Issue #11)
- **File:** `sklearn/discriminant_analysis.py` (lines 1254-1258)
- **Change:** In `partial_fit`, the SVD path raises `ValueError` for `covariance_estimator is not None`. Change to `NotImplementedError` to match the eigen/lsqr path (line 1261-1265). Update the error message to be consistent.
- **File:** `sklearn/tests/test_discriminant_analysis.py`
- **Change:** Update the corresponding test (`test_lda_partial_fit_raises_covariance_estimator` equivalent for SVD) to expect `NotImplementedError`.

### Step 3: Fix Array API overclaim in docstrings/comments (Issue #5)
- **File:** `sklearn/discriminant_analysis.py`
- **Change:** Update the `_partial_fit_svd` docstring (line 867: "Array API compatible: tensors stay on-device") and `_reconstruct_svd_attrs` docstring (line 748: "Array API compatible: operations stay on-device") to accurately state: the SVD path stays on-device when the backend supports float64; eigen/lsqr convert to NumPy. Also update the partial_fit method's inline comments (lines 1280-1284) to be precise.

### Step 4: Fix test parity overclaims in docstrings (Issue #6)
- **File:** `sklearn/tests/test_discriminant_analysis.py`
- **Change:** Review test docstrings for overclaims about "exact equivalence" across all solvers/methods. Ensure docstrings accurately reflect: (a) lsqr doesn't implement `transform`, (b) SVD adversarial tests use approximate tolerance (`atol=1e-5`), (c) batch-equivalence tests use simple synthetic data.

### Step 5: Verify common test suite compliance (Issue #13)
- **Action:** Run `check_estimator(LinearDiscriminantAnalysis())` and document any failures. Fix if needed.

---

## Phase 2: Strongly Recommended Changes

### Step 6: Drop whitened-energy truncation pass 2 (Issue #17 / Strongly Recommended)
- **File:** `sklearn/discriminant_analysis.py` (lines 1006-1048)
- **Change:** Remove the entire Pass 2 whitened-energy truncation block (~40 lines). Keep only Pass 1 (numerical rank thresholding, lines 996-1005). Set `rank = num_rank` directly after Pass 1. This removes uncitable algorithmic logic, a Python loop over singular values, and simplifies review.
- **File:** `sklearn/tests/test_discriminant_analysis.py`
- **Change:** Remove or update any tests that specifically validate whitened-energy truncation behavior.

---

## Phase 3: Should-Fix Issues

### Step 7: Add pickle roundtrip test (Issue #9 sub-item)
- **File:** `sklearn/tests/test_discriminant_analysis.py`
- **Change:** Add a test that pickles and unpickles an LDA model fitted via `partial_fit` with `solver="svd"`, then verifies predictions are identical. This validates the `__getattr__` lazy mechanism survives serialization.

### Step 8: Fix `covariance_` in lazy set when `store_covariance=False` (Issue #9 sub-item)
- **File:** `sklearn/discriminant_analysis.py`
- **Change:** In `__getattr__` (line 701-709), add a guard: if `name == "covariance_"` and `not self.store_covariance`, raise `AttributeError` directly without triggering `_reconstruct_svd_attrs()`. This avoids wasteful computation.

### Step 9: Fix error type for SVD covariance_estimator in batch `fit` (Issue #11 extension)
- **File:** `sklearn/discriminant_analysis.py` (lines 1159-1164)
- **Change:** The batch `fit` SVD path raises `ValueError` for `covariance_estimator`. Change to `NotImplementedError` for consistency.
- **File:** `sklearn/tests/test_discriminant_analysis.py`
- **Change:** Update corresponding test expectations.

### Step 10: Replace MNIST network test with offline alternative (Issue #15)
- **File:** `sklearn/tests/test_discriminant_analysis.py` (line 1817+)
- **Change:** Replace `test_lda_svd_partial_fit_mnist_accuracy_parity` that uses `fetch_openml("mnist_784")` with a test using `sklearn.datasets.load_digits()` (built-in, offline, 64 features, 10 classes) or a synthetic dataset with similar characteristics (zero-variance columns, many features). This eliminates the network dependency and CI flakiness risk.

### Step 11: Rename `_clear_prediction_attrs` vs `_invalidate_svd_attrs` for clarity (Issue #12)
- **File:** `sklearn/discriminant_analysis.py`
- **Change:** Add clear docstrings contrasting the two methods:
  - `_invalidate_svd_attrs`: deletes cached attrs AND marks `_svd_attrs_stale = True` (used after SVD accumulators update, triggers lazy rebuild on next access)
  - `_clear_prediction_attrs`: deletes cached attrs AND sets `_svd_attrs_stale = False` (used during full reinitialization, no lazy rebuild)
- Consider renaming `_clear_prediction_attrs` → `_reset_fitted_attrs` to make the "full reset" semantics clearer.

### Step 12: Add cost-scaling note to docstring (Issue from recommendations)
- **File:** `sklearn/discriminant_analysis.py`
- **Change:** Add a note to the `partial_fit` docstring: "Per-chunk cost for the SVD solver is O((r + m)^2 * d) where r is the stored rank and m is the chunk size. For high-dimensional dense data (d > 10K), the covariance-based solvers (eigen/lsqr) are recommended."

### Step 13: Add Array API justification comments to per-class loops (Issue #10)
- **File:** `sklearn/discriminant_analysis.py` (lines 908, 919, 928, 943)
- **Change:** Add inline comments explaining that the per-class Python loops exist because boolean indexing assignment is not universally supported in the Array API standard. The comment at line 955 already does this for one loop; replicate for the others.

### Step 14: Flag batch-path refactoring explicitly (Issue #14)
- **File:** `sklearn/discriminant_analysis.py`
- **Change:** Add a comment near `_clamp_svd_std` (line 598) noting: "Refactored from inline `std[std == 0] = 1.0` in `_solve_svd` — no behavioral change in batch `fit`."

---

## Phase 4: Verification

### Step 15: Run the full discriminant analysis test suite
- Run `pytest sklearn/tests/test_discriminant_analysis.py -v` to verify all changes pass.

### Step 16: Run common estimator checks
- Run `check_estimator` on `LinearDiscriminantAnalysis` with `partial_fit` to verify compliance.

### Step 17: Commit and push
- Commit all changes with a descriptive message.
- Push to `claude/lda-partial-fit-review-9HSMv`.

---

## Out of Scope (noted for PR description, not implemented here)
- Maintainer engagement / design summary on issue #30042 (process, not code)
- PR description rewrite / AI disclosure placement (PR metadata, not code)
- Offer to split PR (PR comment, not code)
- Benchmark eager vs. lazy reconstruction (requires benchmarking infrastructure)
- Thread safety of `__getattr__` (architectural decision requiring maintainer input)
- Brand citation corrections (PR description text)
- Kim SSSA reference removal (PR description text)
- Vectorizing per-class loops (optimization, separate PR)
