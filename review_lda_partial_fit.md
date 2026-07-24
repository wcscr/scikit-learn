# Maintainer Review: `partial_fit` for `LinearDiscriminantAnalysis`

**Branch:** `wcscr/scikit-learn:feature/lda-partial-fit-clean-withsvd`
**Closes:** #30042
**Diff:** +1,799 lines across 3 files (discriminant_analysis.py, test_discriminant_analysis.py, changelog)

---

## Overall Assessment

This is a substantial, well-tested contribution. The implementation is correct, the test coverage is excellent (31 new tests, 114 total after parametrize expansion), and Array API compatibility is handled thoughtfully for the SVD path. The areas that will draw the hardest review questions are: (1) the `__getattr__` lazy-reconstruction mechanism, (2) the novel whitened-energy truncation strategy in the SVD path, which lacks a clean citation and may be better deferred to a follow-up, (3) the computational cost profile of the SVD path on high-dimensional data, and (4) whether prior maintainer buy-in exists for this approach. In addition, the draft comment contains factual overclaims about Array API support and test coverage that would damage credibility if posted as-is, and at +1,799 lines the PR should offer to be split.

---

## Process / Presentation Issues

### 1. Maintainer engagement

The opening line references an "earlier comment" to @glemaitre, but issue #30042 has no comments from maintainers beyond the original issue body. If that exchange happened elsewhere (Discord, etc.), the context is lost. scikit-learn discourages large unsolicited PRs. Consider posting a short design summary on the issue first and asking for feedback before opening the full PR.

### 2. Comment vs. PR description redundancy

The issue comment and the "Draft PR Description" section largely repeat each other. Post a shorter version on the issue summarizing the approach and asking for review; put the detailed description on the PR itself, where maintainers will be reading the diff.

### 3. Changelog attribution

The changelog uses `:user:\`Will Cobb <willcobb>\``, but the GitHub account is `wcscr`. The Sphinx `:user:` role generates a link to `github.com/willcobb`. Verify this resolves correctly or change to `<wcscr>`.

### 4. Placeholder changelog filename

`99999.feature.rst` — the scikit-learn convention is the PR number. Note in the comment that it will be updated.

### 5. Array API claim is factually wrong

The draft says "Array API compatible across all paths." This is falsifiable in 20 seconds by reading `partial_fit`: the eigen/lsqr paths explicitly call `_convert_to_numpy(X, xp)`, and the SVD path falls back to NumPy on backends lacking float64 (e.g., MPS). There is even a test (`test_lda_svd_partial_fit_array_api_covariance_path_converts`) that verifies the covariance path converts to NumPy. The accurate statement is: "The SVD `partial_fit` path stays on-device when the backend supports float64; eigen/lsqr convert to NumPy." Overclaims like this make a reviewer start auditing everything else, which is the opposite of what you want.

### 6. Test description overstates parity

The draft claims "exact equivalence for `predict`, `transform`, `predict_proba`, and `decision_function` across various chunk sizes." Three problems: (a) `lsqr` does not implement `transform` — it raises `NotImplementedError` — so claiming transform parity across all solvers is wrong; (b) the SVD adversarial tests use `atol=1e-5`, which is approximate parity, not exact; (c) the batch-equivalence tests on simple synthetic data are not the same as parity "across various chunk sizes" in the adversarial sense. Say what the tests actually verify.

### 7. AI disclosure belongs in the PR description, not the issue comment

scikit-learn's contributor guidelines explicitly state that AI disclosure goes in the PR description and discourage AI-generated text in issue/PR comments. Move "This pull request includes code written with the assistance of AI" to the PR body. More broadly, the comment's prose has the over-complete, exhaustive style that signals AI-generated content to experienced reviewers — trim aggressively.

### 8. Offer to split the PR

At +1,799 lines, the diff exceeds what most maintainers will review in one pass. scikit-learn's guidelines recommend smaller PRs. The SVD path is substantially more complex than eigen/lsqr (lazy reconstruction, `__getattr__`, truncation, Array API). Adding a single line — "Happy to split eigen/lsqr and svd into separate PRs if that would make review easier" — makes the whole contribution feel more collaborative and less like a monolith.

---

## Code-Level Issues

### 9. `__getattr__` is the highest-risk mechanism

This is globally active on *all* instances, not just those created via `partial_fit`. Specific concerns:

- **Thread safety:** Once fitted, scikit-learn estimators' `predict` methods should be read-only and safe for concurrent use across threads. If `__getattr__` mutates `__dict__` by caching reconstructed attributes during the first `predict` call, that introduces a write during what should be a pure read — creating a race condition under concurrent inference. This is arguably the strongest objection to the lazy approach.
- **Pickle roundtrip:** During `__setstate__`, `__dict__` may not yet be populated when attribute access occurs. The `self.__dict__.get("_svd_attrs_stale", False)` pattern is safe, but a maintainer will want an explicit pickle roundtrip test.
- **`hasattr` side effects:** Any code calling `hasattr(lda, "coef_")` on a stale SVD instance triggers full reconstruction. The `__sklearn_is_fitted__` override handles the main case, but third-party code or sklearn internals doing `hasattr` on other lazy attrs (e.g., `scalings_`) will still trigger it.
- **`covariance_` in the lazy set:** When `store_covariance=False`, accessing `covariance_` still triggers `_reconstruct_svd_attrs()`, does the full computation, then raises `AttributeError` because `covariance_` was never set. Consider checking `store_covariance` in `__getattr__` or removing `covariance_` from the lazy set.
- **Interaction with `clone` / `get_params`:** These should be safe (they access constructor params only), but a maintainer will ask you to verify.

**The eager-vs-lazy tradeoff is real, not one-sided.** One natural response is to simply compute `coef_`, `intercept_`, `scalings_`, etc., eagerly at the end of every `partial_fit` call. But `_reconstruct_svd_attrs` calls `scipy.linalg.svd` twice (within-class whitened scaling and between-class scaling). For a user streaming hundreds of small chunks where only the final model matters, paying two SVDs per chunk instead of once at the end is a meaningful cost — it's the reason the lazy design exists. The right solution may be eager reconstruction with a guard that skips recomputation when the accumulators haven't changed, or it may be accepting the eager cost and documenting it. Either way, a maintainer will want you to quantify the overhead (benchmark eager vs. lazy on, say, 500 single-sample chunks of MNIST) rather than hand-wave in either direction.

### 10. Python-level loops over classes

`_partial_fit_svd` has four separate `for idx, c in enumerate(self.classes_)` loops for chunk stats, mean-shift vectors, sum-of-squares updates, and mean/count updates. With many classes (e.g., Olivetti has 40), this is slow. scikit-learn prefers vectorized implementations. The whitened-energy truncation loop over singular values has the same concern. Since Array API compatibility motivates the per-class loops (boolean indexing isn't universally supported), say so explicitly in code comments.

### 11. Error type inconsistency

In the SVD path, `covariance_estimator is not None` raises `ValueError`. In the eigen/lsqr path, the same condition raises `NotImplementedError`. Since `partial_fit` with these options is conceptually "not yet implemented" rather than "wrong input," use `NotImplementedError` for both. Additionally, the draft's blanket explanation that unsupported configs "require the full dataset" is too glib for custom covariance estimators — the real reason is that the incremental design doesn't support arbitrary estimator objects, which is a design constraint, not a mathematical one.

### 12. `_clear_prediction_attrs` vs `_invalidate_svd_attrs` naming

Both delete lazy attributes. The distinction (one resets `_svd_attrs_stale = False`, the other sets `True`) is non-obvious. Rename or add docstrings that explicitly contrast them.

### 13. No mention of common test compliance

Does `check_estimator(LinearDiscriminantAnalysis())` pass? The common test suite has specific checks for `partial_fit` (e.g., `check_estimators_partial_fit_n_features`). Confirm this in the PR description. If there are failures, address them before maintainers will review seriously.

### 14. Batch-path behavior change

The `_solve_svd` modification (`std[std == 0] = 1.0` → `self._clamp_svd_std(std)`) alters the batch `fit` code path. While the clamping logic is functionally equivalent, modifying batch behavior in a PR that's ostensibly "adding partial_fit" will draw questions. Flag explicitly: "Refactored std clamping into `_clamp_svd_std` — no behavioral change in batch `fit`."

### 15. MNIST test with network dependency

The MNIST test requires `SKLEARN_SKIP_NETWORK_TESTS=0`. sklearn CI typically skips network tests and is strict about network-caused flakiness. Consider replacing MNIST with `sklearn.datasets.load_digits()` (built-in, offline, still has zero-variance background pixels) or a synthetic dataset from `make_classification` with the specific collinearity/variance characteristics you need. If you strictly need the high dimensionality of MNIST to trigger N < D edge cases, synthesize it offline.

### 16. `versionadded:: 1.9`

Verify this matches the actual target release.

---

## Incremental SVD: Algorithmic Tradeoffs and Citation Accuracy

### What the current implementation does

Each `partial_fit` call builds a block matrix `Z` by stacking the previous compact SVD state `diag(S) @ Vt` (rank r), the class-centered chunk data (m rows), and mean-shift correction rows. It then computes `svd(Z, full_matrices=False)` — a full thin SVD of the (r + m) × d matrix — and truncates via a two-pass scheme:

- **Pass 1 — Numerical rank:** Standard epsilon thresholding (`eps × max(shape) × σ₁`). Well-established, citable (Golub & Van Loan).
- **Pass 2 — Whitened energy:** Weights each component's energy by 1/std² (within-class standard deviation per feature) and drops trailing components whose cumulative whitened contribution falls below `tol`. This is LDA-specific and has no single clean citation. See the detailed analysis below.

The block-matrix SVD itself (as distinct from the truncation scheme) is the main computational cost and the main point of comparison with Brand and Kim.

### Brand (2006): Efficient rank-preserving update

Brand's algorithm avoids the full SVD by projecting new data into the existing right singular subspace and only solving a small SVD in the combined basis:

1. **Project:** Split new rows R into `R_par` (within `span(Vt_old)`) and `R_perp` (orthogonal complement).
2. **Thin QR:** Factor `R_perp = J @ K.T` to get p new basis vectors (p ≤ m, often p << d).
3. **Small SVD:** Express Z in the combined basis `[Vt_old; K.T]` to get a matrix of size (r + m) × (r + p). SVD of this small matrix gives the updated factors.
4. **Basis rotation:** Recover `Vt_new` via matrix multiply against the combined basis.

The expensive SVD shrinks from (r + m) × d to (r + m) × (r + p). In high dimensions, this is a large saving. The current implementation and Brand produce the same mathematical result; the difference is purely computational.

### Kim et al. SSSA: Task-aware compression

The Sufficient Spanning Set Approximation takes a fundamentally different perspective. Instead of maintaining a general-purpose low-rank factorization, SSSA asks: what is the minimal set of vectors needed to reconstruct the *discriminant subspace*? The truncation criterion is tied to Fisher discriminant preservation rather than singular value magnitude, so directions with high discriminant power but modest singular values are retained, while high-energy but class-irrelevant directions can be dropped. The spanning set rank is bounded by O(n_classes) rather than O(data rank).

### Per-chunk cost estimates

Estimated leading-term FLOP counts for three approaches, assuming steady-state stored rank r, chunk size m, p new orthogonal directions per chunk (Brand), and SSSA spanning set rank r_sssa ≈ 2–3× n_classes.

**MNIST** (d=784, N=60K, C=10, chunk=1000, r≈200, p≈10, r_sssa≈40)

| Approach | SVD dimensions | Per-chunk cost | Total (60 chunks) | Relative |
|---|---|---|---|---|
| Block-matrix (current) | 1,210 × 784 | 4.5 GFLOP | 268 GFLOP | 1.0× |
| Brand (2006) | 1,210 × 210 | 2.2 GFLOP | 135 GFLOP | 2.0× faster |
| Kim SSSA | — (rank ≈ 40) | 331 MFLOP | 20 GFLOP | 13.5× faster |

For MNIST the block-matrix SVD costs ~1–2ms per chunk on a modern CPU. All three approaches are effectively instantaneous. The optimization headroom exists but doesn't matter in practice.

**Olivetti Faces** (d=4,096, N=400, C=40, chunk=10, r≈150, p≈2, r_sssa≈80)

| Approach | SVD dimensions | Per-chunk cost | Total (40 chunks) | Relative |
|---|---|---|---|---|
| Block-matrix (current) | 200 × 4,096 | 983 MFLOP | 39 GFLOP | 1.0× |
| Brand (2006) | 200 × 152 | 288 MFLOP | 12 GFLOP | 3.4× faster |
| Kim SSSA | — (rank ≈ 80) | 93 MFLOP | 3.7 GFLOP | 10.5× faster |

The N < D edge case. Costs are still modest — under a second total in all cases. The ill-conditioning (already documented in the PR) matters more than the compute.

**20 Newsgroups TF-IDF** (d=130,107, N=18.8K, C=20, chunk=500, r≈800, p≈20, r_sssa≈60)

| Approach | SVD dimensions | Per-chunk cost | Total (38 chunks) | Relative |
|---|---|---|---|---|
| Block-matrix (current) | 1,320 × 130,107 | 1.4 TFLOP | 52 TFLOP | 1.0× |
| Brand (2006) | 1,320 × 820 | 295 GFLOP | 11 TFLOP | 4.6× faster |
| Kim SSSA | — (rank ≈ 60) | 9.2 GFLOP | 349 GFLOP | 148× faster |

This is where the gap becomes material. The block-matrix approach pays seconds per `partial_fit` call on CPU. However, the SVD solver is rarely the right choice at d=130K; users would typically use lsqr or eigen with shrinkage, both of which are handled by the covariance-based `partial_fit` path.

### Recommendations for the PR

**Simplify the SVD truncation.** Drop the whitened-energy pass (pass 2) entirely and ship with numerical-rank-only truncation. This eliminates ~30 lines of uncitable algorithmic logic, removes a Python loop over singular values, and removes a review obstacle — all while preserving mathematical correctness. If rank growth is demonstrated as a practical problem on real workloads, add task-aware truncation in a focused follow-up PR with benchmarks. See the detailed analysis in the whitened-energy truncation section below.

**Be upfront about cost scaling.** Add a note to the docstring or the PR description: "Per-chunk cost is O((r + m)² × d) where r is the stored rank. For high-dimensional dense data (d > 10K) with many incremental updates, the covariance-based solvers (eigen/lsqr) are recommended."

**Fix the citations.** The current PR description cites Brand (2006) as a reference for the SVD update structure. This is defensible as conceptual background — both approaches stack old factors with new data — but the PR should make explicit that the efficient projection/QR trick from Brand is *not* used. The block-matrix approach was chosen for Array API compatibility (no thin QR or subspace tracking required) and implementation simplicity.

Drop the Kim SSSA reference unless the description explains what SSSA is and how the implementation relates. Kim's work informed the idea of task-aware truncation, but the whitened-energy criterion in the code is a different (and less aggressive) heuristic — and if the recommendation to drop that criterion is followed, the Kim connection disappears entirely.

**If a maintainer asks "why not Brand?"**, the answer is: Brand's projection/QR trick requires thin QR (not universally available in array backends) and explicit basis bookkeeping. The 2–5× speedup it provides matters only for d > ~10K, a regime where the SVD solver is already not the recommended choice. Brand's optimization could be added later without changing the public API. The much larger wins from SSSA-style task-aware truncation (up to 148× for high-D data) would require a substantially different algorithmic design — also a valid future direction that doesn't affect the API.

### Whitened-energy truncation: reviewability risk

The two-pass truncation scheme in `_partial_fit_svd` — first numerical-rank thresholding, then cumulative whitened-energy truncation weighted by inverse within-class variance — is the most reviewable-risk algorithmic element in the PR, separate from the `__getattr__` mechanism. The concern is not correctness but *citability and necessity*.

**The review culture problem.** scikit-learn is conservative about algorithmic novelty. When you add a well-known method (`partial_fit` with Chan's pairwise merge), the burden of proof is low — point to the paper, show batch equivalence, done. When you introduce something without a clean citation, maintainers will ask two questions in this order: *"Is this necessary?"* and *"Can this be replaced with something standard?"*

**Is it necessary?** Probably not for the initial PR. Numerical rank truncation (pass 1 alone) is sufficient for correctness. The whitened-energy pass is an optimization that bounds the stored rank more aggressively when data has high numerical rank but low discriminant rank. Without it, the stored rank grows larger, the block matrix gets bigger, and `partial_fit` gets slower over many chunks — but the final model is mathematically the same. A maintainer will look at ~30 lines of loop-based uncitable truncation logic and reasonably say: *"This is premature optimization with a novel heuristic. Drop it. If someone demonstrates a real-world case where rank growth is a problem, we can add smarter truncation in a follow-up."*

**Can it be cited?** Not cleanly. The individual components are standard:

- **Numerical rank thresholding** (pass 1): Golub & Van Loan, *Matrix Computations*, 4th ed., §2.5.2, using `eps × max(shape) × σ₁`.
- **Cumulative energy truncation**: Jolliffe, *Principal Component Analysis*, 2nd ed., §6.1, "proportion of variance explained" criterion.
- **The whitening weight** (1/std² per feature using within-class standard deviation): specific to the LDA context and does not appear in Brand, Kim, or the standard incremental SVD literature.

The combination is novel as applied here. The closest related work is Shental et al.'s "Relevant Component Analysis" (2002), which applies task-specific feature weighting before dimensionality reduction, but it's a loose connection and over-citing it would be misleading.

**If you keep it, there is a defensible framing.** The whitened-energy truncation is more natural than it appears at first glance. `_reconstruct_svd_attrs` operates in whitened space — it computes `SVt / std` and then takes the SVD of *that*. The whitened-energy truncation is therefore just pre-applying the same coordinate transform that reconstruction will apply, and dropping components that will be negligible after that transform. It's the standard cumulative-energy criterion (Jolliffe) applied in the natural coordinate system for the downstream computation, not a novel heuristic. The code comment could read:

> *"Pass 2 truncates in the whitened coordinate system used by reconstruction (§ `_reconstruct_svd_attrs`). Components that contribute < tol to the cumulative energy after whitening by 1/std will be negligible in the final `scalings_` and can be discarded to bound the stored rank. This is the standard proportion-of-variance criterion [Jolliffe, 2002, §6.1] applied in whitened space."*

That's citable, accurate, and doesn't overclaim. But a maintainer may still ask for empirical evidence (benchmarks showing rank growth without it, accuracy impact with it) before accepting it.

**Recommendation: drop it for the initial PR.** Simplify to numerical-rank-only truncation (pass 1). The PR is already large (744 new lines in `discriminant_analysis.py`) and complex enough with the `__getattr__` mechanism, three solver paths, and Array API compatibility. Every line of novel algorithmic logic is a line that slows review and creates an objection surface. Ship the simple correct version, demonstrate it works, and if rank growth turns out to be a practical problem, add whitened-energy truncation as a focused follow-up PR where it gets proper attention, benchmarking, and a clear justification grounded in measured rank growth on real datasets. This also removes the per-singular-value Python loop flagged in issue #10 above, simplifying the code further.

---

## Summary of Action Items

| Priority | Item |
|---|---|
| **Must fix** | Get maintainer buy-in before opening the PR |
| **Must fix** | Fix Array API overclaim — eigen/lsqr convert to NumPy; SVD falls back on backends without float64. The draft's "Array API compatible across all paths" is falsifiable in 20 seconds. |
| **Must fix** | Fix test parity overclaim — lsqr doesn't implement `transform`; SVD adversarial tests use `atol=1e-5` (approximate, not exact) |
| **Must fix** | Move AI disclosure to PR description; scikit-learn guidelines prohibit AI-generated text in issue/PR comments |
| **Must fix** | Verify common test suite passes (`check_estimator`) |
| **Must fix** | Fix changelog attribution (`willcobb` vs `wcscr`) |
| **Must fix** | Fix error type inconsistency (ValueError vs NotImplementedError) |
| **Strongly recommended** | Drop whitened-energy truncation (pass 2); ship with numerical-rank-only. Removes ~30 lines of uncitable logic, a Python loop, and a major review obstacle. Add back in a focused follow-up with benchmarks if rank growth is demonstrated. |
| **Strongly recommended** | Offer to split the PR (eigen/lsqr vs svd). At +1,799 lines, the diff exceeds what most maintainers will review in one pass. |
| **Strongly recommended** | Rewrite the issue comment to ~20 lines: scope, API contract, two review magnets, offer to split. Move everything else to the PR body. |
| **Should fix** | Add pickle roundtrip test for lazy SVD reconstruction |
| **Should fix** | Benchmark eager vs. lazy reconstruction cost (e.g., 500 single-sample chunks on MNIST) to justify whichever approach you choose |
| **Should fix** | If keeping `__getattr__`: address thread safety (concurrent `predict` calls mutate `__dict__` on first access) |
| **Should fix** | Fix `covariance_` in lazy set (wasteful when `store_covariance=False`) |
| **Should fix** | Correct Brand citation — acknowledge no projection trick is used |
| **Should fix** | Drop Kim SSSA reference (connection disappears if whitened truncation is removed) |
| **Should fix** | Add cost-scaling note to docstring for high-D guidance |
| **Should fix** | Replace MNIST network test with `load_digits` or synthetic offline dataset |
| **Should fix** | Rename `_clear_prediction_attrs` vs `_invalidate_svd_attrs` for clarity |
| **Nice to have** | Vectorize per-class loops (or add Array API justification comments) |
| **Nice to have** | Add diagnostic assertion that stored rank stays bounded on MNIST |
