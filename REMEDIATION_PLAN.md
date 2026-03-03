# Plan: Stabilize and Complete `LinearDiscriminantAnalysis.partial_fit` (SVD + Covariance Paths)

## Summary
This plan fixes the five identified deficiencies by making streaming SVD numerically scale-stable, preventing stale/invalid fitted state after early-return paths, enforcing class-observation readiness semantics consistently, restoring `store_covariance` contract consistency, and adding regression tests and docs/changelog updates required for scikit-learn contribution quality.

## Public API / Behavior Changes
1. `LinearDiscriminantAnalysis.partial_fit` will treat the estimator as **not fitted** until all classes declared in `classes=` have been observed at least once.
2. For `solver="svd"` with `store_covariance=True`, `partial_fit` will populate `covariance_` once the estimator becomes fitted, matching `fit` behavior.
3. `partial_fit` after a prior `fit` will no longer reuse stale coefficients when insufficient new data is provided; it will cleanly remain unfitted instead of crashing or silently using stale params.

## Implementation Plan

1. Update SVD streaming numerical thresholds in [discriminant_analysis.py](/Users/willcobb/Code.local/scikit-learn-inclda/sklearn/discriminant_analysis.py).
- Replace absolute singular-value truncation `keep = S_new > self.tol` with relative machine-precision rank filtering:
  `keep = S_new > (eps * max(Z.shape) * S_new[0])` when `S_new.size > 0`; otherwise keep empty.
- Keep `self.tol` only for the final discriminant-space rank choices (aligned with batch solver semantics).
- Replace `std[std < self.tol] = 1.0` with a scale-relative floor:
  compute `std_max = std.max()`; if `std_max == 0`, set all to `1.0`; else `std_floor = sqrt(eps) * std_max` and set `std[std <= std_floor] = 1.0`.
- Apply the same std-floor logic in both `_reconstruct_svd_attrs` and any SVD-reconstruction branch that computes `std`.

2. Add explicit fitted-state invalidation for partial-fit reinitialization in [discriminant_analysis.py](/Users/willcobb/Code.local/scikit-learn-inclda/sklearn/discriminant_analysis.py).
- Introduce a private helper that removes stale prediction attributes:
  `coef_`, `intercept_`, `scalings_`, `xbar_`, `explained_variance_ratio_`, `covariance_`, `_n_features_out`.
- On streaming reinit (`first_call` or missing streaming accumulators after prior `fit`), call that helper before new accumulation starts.
- For SVD path, ensure `_svd_attrs_stale` is reset to `False` on reinit, and only set to `True` via `_invalidate_svd_attrs()` when enough statistics exist to reconstruct prediction attrs.

3. Enforce class-readiness gating consistently in both partial-fit solvers in [discriminant_analysis.py](/Users/willcobb/Code.local/scikit-learn-inclda/sklearn/discriminant_analysis.py).
- Change readiness condition from “at least 2 classes seen” to “all declared classes seen”:
  `n_classes_seen < n_classes` is an early return condition for both `_partial_fit_svd` and `_partial_fit_covariance`.
- Keep existing additional dof check for eigen: `within_df < n_features`.
- Preserve existing unknown-label validation logic.

4. Fix binary-collapse crash after early return in covariance path in [discriminant_analysis.py](/Users/willcobb/Code.local/scikit-learn-inclda/sklearn/discriminant_analysis.py).
- Guard binary collapse in `partial_fit` with shape-aware condition:
  run collapse only when `hasattr(self, "coef_")` and `self.coef_.shape[0] == len(self.classes_) == 2`.
- This prevents `IndexError` after reinit + early return when stale 1-row coefficients were previously present.

5. Restore `store_covariance` contract for SVD partial-fit in [discriminant_analysis.py](/Users/willcobb/Code.local/scikit-learn-inclda/sklearn/discriminant_analysis.py).
- In SVD reconstruction, when `self.store_covariance` is `True`, materialize:
  `covariance_ = (SVt.T @ SVt) / N_total` where `SVt = self._unscaled_S[:, None] * self._unscaled_Vt`.
- Do not materialize covariance when `store_covariance=False` (keep memory behavior unchanged).

6. Update docs/changelog to match behavior.
- Update partial-fit Notes and relevant attribute text in [discriminant_analysis.py](/Users/willcobb/Code.local/scikit-learn-inclda/sklearn/discriminant_analysis.py) docstrings to reflect “all classes must be seen” readiness.
- Update upcoming change note in [99999.feature.rst](/Users/willcobb/Code.local/scikit-learn-inclda/doc/whats_new/upcoming_changes/sklearn.discriminant_analysis/99999.feature.rst) to include SVD support and readiness semantics.
- Ensure `store_covariance` wording is consistent with new partial-fit SVD behavior.

## Test Plan

1. Extend/adjust LDA partial-fit tests in [test_discriminant_analysis.py](/Users/willcobb/Code.local/scikit-learn-inclda/sklearn/tests/test_discriminant_analysis.py).
- Replace SVD no-covariance test with `store_covariance=True` covariance presence + numerical match to batch pooled covariance.
- Add regression test: `fit` then `partial_fit` on one-class chunk does not crash and leaves estimator unfitted (`predict` raises `NotFittedError`) for `lsqr`, `eigen`, and `svd`.
- Add regression test: multiclass (`classes=[0,1,2]`) with only 2 classes observed remains unfitted for all solvers; becomes fitted after missing class arrives.
- Add SVD scale-invariance regression: compare batch vs chunked for global feature scales `[1.0, 1e-4, 1e-6]` with fixed chunking; verify no prediction mismatch and bounded probability error.
- Add near-zero-variance-feature regression: append tiny-noise/constant-like columns; verify batch/chunked parity stays within tolerance and no catastrophic coefficient blow-up.

2. Keep existing parity tests and tighten intent.
- Existing batch-equivalence/transform/predict tests remain; retain sign-alignment logic for transform comparisons.
- Ensure tolerances are dtype-aware:
  float64 parity target around `1e-10`-`1e-12`,
  float32 parity target around `1e-5`-`1e-6`.

3. Validation commands for review gate.
- `pytest -q sklearn/tests/test_discriminant_analysis.py`
- `pytest -q sklearn/tests/test_discriminant_analysis.py -k "partial_fit or svd"`
- `python - <<'PY' ... check_estimator(LinearDiscriminantAnalysis()) ... PY`
- If Array API deps are available in CI/dev env, run the corresponding array-api estimator checks for LDA.

## Acceptance Criteria
1. No `IndexError` in `partial_fit` after a prior `fit` across solvers.
2. No stale predictions after early-return reinit paths; estimator correctly reports unfitted until ready.
3. SVD chunked vs batch remains prediction-equivalent across tested chunk sizes and global scales.
4. `store_covariance=True` with SVD partial-fit exposes `covariance_` and matches batch pooled covariance within tolerance.
5. Docs/changelog accurately describe solver support and fitted-readiness semantics.
6. Updated tests pass locally and in CI.

## Assumptions and Defaults Chosen
1. Chosen policy: require all declared classes to be observed before `partial_fit` model is considered fitted.
2. Chosen policy: for `solver="svd"` + `store_covariance=True`, materialize `covariance_` after readiness.
3. No new public parameters will be introduced; all fixes are behavior/documentation consistency and numerical-stability improvements.

This pull request includes code written with the assistance of AI.  
The code has **not yet been reviewed** by a human.
