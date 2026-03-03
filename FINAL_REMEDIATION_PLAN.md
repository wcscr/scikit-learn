# Plan: Close Remaining SVD `partial_fit` Parity Gap and Harden Regression Coverage

## Summary
Two issues remain:
1. A real parity bug is still reproducible for multiclass data with exactly constant columns in the `make_classification` regime.
2. The newly added regression test is valid but too narrow to guard the full failure surface.

This plan fixes the implementation in a minimal, deterministic way and adds a broad adversarial test that reproduces the current failure before the fix and must pass after.

## Public API / Interface Changes
No public API changes.  
No parameter changes.  
Only numerical-stability internals and tests are updated.

## Step-by-Step Implementation Plan

1. Add a failing adversarial regression test that matches the known failing regime.
- File: [test_discriminant_analysis.py](/Users/willcobb/Code.local/scikit-learn-inclda/sklearn/tests/test_discriminant_analysis.py)
- New test name: `test_lda_svd_partial_fit_multiclass_constant_columns_make_classification`
- Dataset:
  - `make_classification(n_samples=700, n_features=12, n_informative=8, n_redundant=0, n_classes=3, n_clusters_per_class=1, random_state=0)`
  - Append 2 exactly constant columns (for example `3.14` and `3.14`)
- Evaluate chunk sizes: `[1, 2, 7, 64, 127, 700]`
- Assertions per chunk size:
  - `assert_array_equal(online.predict(X), batch.predict(X))`
  - `assert_allclose(online.predict_proba(X), batch.predict_proba(X), atol=1e-10)`

2. Introduce one shared helper for SVD within-class `std` clamping at machine-noise scale.
- File: [discriminant_analysis.py](/Users/willcobb/Code.local/scikit-learn-inclda/sklearn/discriminant_analysis.py)
- Add private helper on `LinearDiscriminantAnalysis`, for example `_clamp_svd_std(std)`:
  - Compute `std_max = float(np.max(std))` (or namespace equivalent converted to Python float).
  - Compute `noise_floor = np.finfo(np.float64).eps * max(std_max, 1.0)`.
  - Replace `std <= noise_floor` with `1.0`.
- Rationale:
  - Clamps numerical fuzz (`~1e-16`) that causes huge coefficient blow-ups in batch.
  - Preserves true tiny-but-nonzero signals (`1e-8`, `1e-9`) since they are far above `noise_floor`.
  - Avoids the over-aggressive `sqrt(eps)` floor that previously broke tiny-noise parity.

3. Apply the helper in both batch and streaming SVD paths.
- Batch path: `_solve_svd`, right after `std = xp.std(Xc, axis=0)` and before scaling.
- Streaming path: `_reconstruct_svd_attrs`, right after reconstructing `std`.
- Remove direct ad-hoc clamp expressions (`std == 0` etc.) and use only the shared helper so semantics remain aligned.

4. Keep current relative rank filtering unchanged.
- Do not alter `keep = S_new > eps * max(Z.shape) * S_new[0]` in `_partial_fit_svd`.
- The parity issue here is from inconsistent `std` floor semantics, not from rank filtering policy itself.

5. Expand test coverage beyond the current narrow synthetic case.
- Keep your existing new test `test_lda_svd_partial_fit_multiclass_constant_columns`.
- Add one more variant test for robustness:
  - Same `make_classification` setup.
  - Exactly constant columns.
  - Chunk size worst-case `1`.
  - Exact prediction and probability parity checks.
- This ensures both synthetic and realistic generated distributions are covered.

6. Validate locally with focused and full runs.
- Focused:
  - `pytest -q sklearn/tests/test_discriminant_analysis.py -k "svd_partial_fit and (constant_columns or tiny_nonzero_noise or scale_invariance)"`
- Full module:
  - `pytest -q sklearn/tests/test_discriminant_analysis.py`
- Optional estimator check:
  - `python - <<'PY'\nfrom sklearn.utils.estimator_checks import check_estimator\nfrom sklearn.discriminant_analysis import LinearDiscriminantAnalysis\ncheck_estimator(LinearDiscriminantAnalysis())\nprint('ok')\nPY`

## Adversarial Verification Script
Use this script before and after implementation.  
Expected now (pre-fix): at least one chunk size fails.  
Expected after fix: all chunk sizes pass.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def run_case(chunk_size: int):
    X, y = make_classification(
        n_samples=700,
        n_features=12,
        n_informative=8,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=0,
    )
    X = X.astype(np.float64)
    X = np.hstack([X, np.ones((X.shape[0], 2), dtype=np.float64) * 3.14])

    classes = np.unique(y)

    batch = LinearDiscriminantAnalysis(solver="svd").fit(X, y)

    online = LinearDiscriminantAnalysis(solver="svd")
    for i in range(0, X.shape[0], chunk_size):
        online.partial_fit(
            X[i : i + chunk_size],
            y[i : i + chunk_size],
            classes=classes if i == 0 else None,
        )

    pred_equal = np.array_equal(online.predict(X), batch.predict(X))
    proba_err = float(np.max(np.abs(online.predict_proba(X) - batch.predict_proba(X))))
    coef_err = float(np.max(np.abs(online.coef_ - batch.coef_)))
    return pred_equal, proba_err, coef_err


if __name__ == "__main__":
    chunk_sizes = [1, 2, 7, 64, 127, 700]
    failed = False
    for cs in chunk_sizes:
        pred_equal, proba_err, coef_err = run_case(cs)
        print(
            f"chunk={cs:>3} pred_equal={pred_equal} "
            f"max_proba_err={proba_err:.3e} max_coef_err={coef_err:.3e}"
        )
        if not pred_equal or proba_err > 1e-10:
            failed = True

    if failed:
        raise SystemExit("FAIL: batch/streaming parity check failed.")
    print("PASS: all chunk sizes satisfy parity.")
```

## Acceptance Criteria
1. The adversarial script above passes for all listed chunk sizes.
2. New `make_classification + constant columns` regression test passes.
3. Existing strict parity tests (`tiny_nonzero_noise`, `scale_invariance`) still pass.
4. Full discriminant-analysis test module passes without regressions.

## Assumptions and Defaults
1. Default decision: treat only machine-noise-level `std` values as zero using `eps * max(std, 1.0)` with float64 epsilon.
2. Default decision: keep existing SVD rank-filtering logic unchanged.
3. Default decision: keep all public behavior and APIs unchanged except improved numerical consistency.

This pull request includes code written with the assistance of AI.
The code has **not yet been reviewed** by a human.
