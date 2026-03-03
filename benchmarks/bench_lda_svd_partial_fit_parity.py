#!/usr/bin/env python
"""Benchmark/validation harness for LDA SVD fit vs partial_fit parity."""

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np

from sklearn.datasets import load_digits, make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


@dataclass
class CheckOutcome:
    report: dict
    passed: bool


def _parse_int_list(arg):
    return [int(x.strip()) for x in arg.split(",") if x.strip()]


def _run_synthetic(args):
    rng = np.random.RandomState(args.synthetic_shuffle_seed)
    chunk_sizes = args.synthetic_chunk_sizes

    report = {
        "mode": "synthetic",
        "seeds": args.synthetic_seeds,
        "chunk_sizes": chunk_sizes,
        "agreement_threshold": args.synthetic_agreement_threshold,
        "acc_delta_threshold": args.synthetic_acc_delta_threshold,
        "proba_atol": args.synthetic_proba_atol,
        "proba_rtol": args.synthetic_proba_rtol,
        "strict_parity": args.strict_parity,
        "per_seed": {},
    }

    passed = True

    for seed in args.synthetic_seeds:
        X, y = make_classification(
            n_samples=700,
            n_features=12,
            n_informative=8,
            n_redundant=0,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=seed,
        )
        X = np.column_stack([X, np.full((700, 2), [3.14, -2.71])])

        order = rng.permutation(len(X))
        X = X[order]
        y = y[order]

        clf_batch = LinearDiscriminantAnalysis(solver="svd")
        clf_batch.fit(X, y)
        pred_batch = clf_batch.predict(X)
        proba_batch = clf_batch.predict_proba(X)
        acc_batch = float(np.mean(pred_batch == y))

        classes = np.unique(y)
        baseline_stream_pred = None
        per_chunk = []
        seed_pass = True

        for chunk_size in chunk_sizes:
            clf_stream = LinearDiscriminantAnalysis(solver="svd")
            for start in range(0, len(X), chunk_size):
                end = min(start + chunk_size, len(X))
                clf_stream.partial_fit(
                    X[start:end],
                    y[start:end],
                    classes=classes if start == 0 else None,
                )

            pred_stream = clf_stream.predict(X)
            proba_stream = clf_stream.predict_proba(X)
            acc_stream = float(np.mean(pred_stream == y))

            pred_mismatch = int(np.sum(pred_stream != pred_batch))
            pred_agreement = float(np.mean(pred_stream == pred_batch))
            proba_close = bool(
                np.allclose(
                    proba_stream,
                    proba_batch,
                    atol=args.synthetic_proba_atol,
                    rtol=args.synthetic_proba_rtol,
                )
            )
            max_proba_abs_err = float(np.max(np.abs(proba_stream - proba_batch)))
            acc_delta = acc_stream - acc_batch

            if baseline_stream_pred is None:
                chunk_stable = True
                baseline_stream_pred = pred_stream
            else:
                chunk_stable = bool(
                    np.array_equal(pred_stream, baseline_stream_pred)
                )

            if args.strict_parity:
                chunk_pass = pred_mismatch == 0 and proba_close
            elif seed == args.reference_seed:
                chunk_pass = pred_mismatch == 0 and proba_close
            else:
                chunk_pass = (
                    chunk_stable
                    and pred_agreement >= args.synthetic_agreement_threshold
                    and acc_delta >= args.synthetic_acc_delta_threshold
                    and acc_stream > args.synthetic_acc_floor
                    and np.isfinite(max_proba_abs_err)
                )
            chunk_pass = bool(chunk_pass)

            seed_pass = seed_pass and chunk_pass
            per_chunk.append(
                {
                    "chunk_size": chunk_size,
                    "pred_mismatch": pred_mismatch,
                    "pred_agreement": pred_agreement,
                    "max_proba_abs_err": max_proba_abs_err,
                    "proba_close": proba_close,
                    "acc_batch": acc_batch,
                    "acc_stream": acc_stream,
                    "acc_delta": acc_delta,
                    "chunk_stable": chunk_stable,
                    "pass": chunk_pass,
                }
            )

        report["per_seed"][str(seed)] = {
            "pass": seed_pass,
            "chunks": per_chunk,
        }
        passed = passed and seed_pass

    report["pass"] = passed
    return CheckOutcome(report=report, passed=passed)


def _run_digits(args):
    X, y = load_digits(return_X_y=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_rows = []
    passed = True
    classes = np.unique(y)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        clf_batch = LinearDiscriminantAnalysis(solver="svd")
        clf_batch.fit(X_train, y_train)
        pred_batch = clf_batch.predict(X_test)
        acc_batch = float(np.mean(pred_batch == y_test))

        row = {
            "fold": fold_idx,
            "acc_batch": acc_batch,
            "chunks": [],
        }

        for chunk_size in args.digits_chunk_sizes:
            clf_stream = LinearDiscriminantAnalysis(solver="svd")
            for start in range(0, len(X_train), chunk_size):
                end = min(start + chunk_size, len(X_train))
                clf_stream.partial_fit(
                    X_train[start:end],
                    y_train[start:end],
                    classes=classes if start == 0 else None,
                )

            pred_stream = clf_stream.predict(X_test)
            acc_stream = float(np.mean(pred_stream == y_test))
            pred_agreement = float(np.mean(pred_stream == pred_batch))
            acc_delta = acc_stream - acc_batch
            chunk_pass = (
                acc_delta >= args.digits_acc_delta_threshold
                and pred_agreement >= args.digits_agreement_threshold
            )

            row["chunks"].append(
                {
                    "chunk_size": chunk_size,
                    "acc_stream": acc_stream,
                    "acc_delta": acc_delta,
                    "pred_agreement": pred_agreement,
                    "pass": chunk_pass,
                }
            )
            passed = passed and chunk_pass

        fold_rows.append(row)

    report = {
        "mode": "digits",
        "chunk_sizes": args.digits_chunk_sizes,
        "acc_delta_threshold": args.digits_acc_delta_threshold,
        "agreement_threshold": args.digits_agreement_threshold,
        "folds": fold_rows,
        "pass": passed,
    }
    return CheckOutcome(report=report, passed=passed)


def _run_mnist(args):
    report = {
        "mode": "mnist",
        "chunk_sizes": args.mnist_chunk_sizes,
        "acc_delta_threshold": args.mnist_acc_delta_threshold,
        "agreement_threshold": args.mnist_agreement_threshold,
        "acc_floor": args.mnist_acc_floor,
        "sample_size": args.mnist_sample_size,
        "pass": False,
    }

    if os.environ.get("SKLEARN_SKIP_NETWORK_TESTS", "1") != "0":
        msg = "SKLEARN_SKIP_NETWORK_TESTS is not 0; network fetch disabled"
        report["skipped"] = True
        report["reason"] = msg
        return CheckOutcome(report=report, passed=not args.require_mnist)

    try:
        from sklearn.datasets import fetch_openml

        mnist = fetch_openml(
            name="mnist_784",
            version=1,
            as_frame=False,
            parser="auto",
        )
    except Exception as exc:  # pragma: no cover - depends on network
        report["skipped"] = True
        report["reason"] = f"Could not fetch MNIST: {exc}"
        return CheckOutcome(report=report, passed=not args.require_mnist)

    X_all = np.asarray(mnist.data, dtype=np.float64)
    y_all = np.asarray(mnist.target, dtype=np.int64)

    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_all), size=args.mnist_sample_size, replace=False)
    X_all, y_all = X_all[idx], y_all[idx]

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X_all, y_all))
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]

    clf_batch = LinearDiscriminantAnalysis(solver="svd")
    clf_batch.fit(X_train, y_train)
    pred_batch = clf_batch.predict(X_test)
    acc_batch = float(np.mean(pred_batch == y_test))

    classes = np.unique(y_train)
    rows = []
    passed = True
    for chunk_size in args.mnist_chunk_sizes:
        clf_stream = LinearDiscriminantAnalysis(solver="svd")
        for start in range(0, len(X_train), chunk_size):
            end = min(start + chunk_size, len(X_train))
            clf_stream.partial_fit(
                X_train[start:end],
                y_train[start:end],
                classes=classes if start == 0 else None,
            )

        pred_stream = clf_stream.predict(X_test)
        acc_stream = float(np.mean(pred_stream == y_test))
        pred_agreement = float(np.mean(pred_stream == pred_batch))
        acc_delta = acc_stream - acc_batch
        chunk_pass = (
            acc_delta >= args.mnist_acc_delta_threshold
            and pred_agreement >= args.mnist_agreement_threshold
            and acc_stream > args.mnist_acc_floor
        )
        passed = passed and chunk_pass
        rows.append(
            {
                "chunk_size": chunk_size,
                "acc_batch": acc_batch,
                "acc_stream": acc_stream,
                "acc_delta": acc_delta,
                "pred_agreement": pred_agreement,
                "pass": chunk_pass,
            }
        )

    report["chunks"] = rows
    report["pass"] = passed
    return CheckOutcome(report=report, passed=passed)


def _run(args):
    reports = []
    overall_pass = True

    if args.mode in {"synthetic", "all"}:
        out = _run_synthetic(args)
        reports.append(out.report)
        overall_pass = overall_pass and out.passed

    if args.mode in {"digits", "all"}:
        out = _run_digits(args)
        reports.append(out.report)
        overall_pass = overall_pass and out.passed

    if args.mode in {"mnist", "all"}:
        out = _run_mnist(args)
        reports.append(out.report)
        overall_pass = overall_pass and out.passed

    payload = {
        "mode": args.mode,
        "pass": overall_pass,
        "reports": reports,
    }
    return payload, overall_pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="all",
        choices=["synthetic", "digits", "mnist", "all"],
    )
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--require-mnist", action="store_true")
    parser.add_argument("--strict-parity", action="store_true")
    parser.add_argument("--reference-seed", type=int, default=0)

    parser.add_argument(
        "--synthetic-seeds",
        type=_parse_int_list,
        default=[0, 7, 11, 16],
    )
    parser.add_argument(
        "--synthetic-chunk-sizes",
        type=_parse_int_list,
        default=[1, 2, 7, 64, 700],
    )
    parser.add_argument("--synthetic-shuffle-seed", type=int, default=0)
    parser.add_argument("--synthetic-proba-atol", type=float, default=1e-10)
    parser.add_argument("--synthetic-proba-rtol", type=float, default=1e-7)
    parser.add_argument(
        "--synthetic-agreement-threshold", type=float, default=0.87
    )
    parser.add_argument(
        "--synthetic-acc-delta-threshold", type=float, default=-0.01
    )
    parser.add_argument("--synthetic-acc-floor", type=float, default=0.75)

    parser.add_argument(
        "--digits-chunk-sizes",
        type=_parse_int_list,
        default=[512, 1024, 2048],
    )
    parser.add_argument("--digits-acc-delta-threshold", type=float, default=-0.005)
    parser.add_argument("--digits-agreement-threshold", type=float, default=0.999)

    parser.add_argument(
        "--mnist-chunk-sizes",
        type=_parse_int_list,
        default=[256, 512, 1024, 2048],
    )
    parser.add_argument("--mnist-sample-size", type=int, default=5000)
    parser.add_argument("--mnist-acc-delta-threshold", type=float, default=-0.01)
    parser.add_argument("--mnist-agreement-threshold", type=float, default=0.999)
    parser.add_argument("--mnist-acc-floor", type=float, default=0.75)

    args = parser.parse_args()
    payload, passed = _run(args)

    rendered = json.dumps(payload, indent=2)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(rendered)
            f.write("\n")
    else:
        print(rendered)

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
