#!/usr/bin/env python
"""Linear-decodability audit (F7 §1b).

Fits a ridge probe from the **raw input window** to every candidate probe target and reports R² on
the test split. Any target the raw window already solves is disqualified as a headline E1 target,
because "the latent recovered it" would then be a statement about window geometry rather than about
the representation.

This runs before any model exists, and should be re-run whenever the window configuration changes.

    python scripts/linear_audit.py --dataset data/windows_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flatjepa.data.dataset import WindowedDataset
from flatjepa.data.targets import TARGET_SPECS

DEFAULT_THRESHOLD = 0.9


def ridge_fit_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float
) -> np.ndarray:
    """Closed-form ridge with an unpenalised intercept."""
    xtr = np.concatenate([x_train, np.ones((len(x_train), 1))], axis=1)
    xte = np.concatenate([x_test, np.ones((len(x_test), 1))], axis=1)
    gram = xtr.T @ xtr + alpha * np.eye(xtr.shape[1])
    gram[-1, -1] -= alpha  # do not penalise the intercept
    weights = np.linalg.solve(gram, xtr.T @ y_train)
    return xte @ weights


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean per-channel R². Channels with no variance contribute 0 rather than a divide-by-zero."""
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    per_channel = np.where(ss_tot > 1e-12, 1.0 - ss_res / np.maximum(ss_tot, 1e-12), 0.0)
    return float(np.mean(per_channel))


def audit(
    dataset_root: str | Path,
    alpha: float = 1e-3,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    train = WindowedDataset(dataset_root, "train")
    test = WindowedDataset(dataset_root, "test")
    x_train, x_test = train.flat_inputs(), test.flat_inputs()

    results = {}
    for spec in TARGET_SPECS:
        y_train, y_test = train.target(spec.name), test.target(spec.name)
        pred = ridge_fit_predict(x_train, y_train, x_test, alpha)
        r2 = r2_score(y_test, pred)
        results[spec.name] = {
            "r2_raw_window": r2,
            "expected_kind": spec.kind,
            "disqualified": bool(r2 > threshold),
            "width": spec.width,
        }

    return {
        "dataset": str(dataset_root),
        "n_train": len(train),
        "n_test": len(test),
        "input_dim": int(x_train.shape[1]),
        "alpha": alpha,
        "threshold": threshold,
        "targets": results,
    }


def format_report(report: dict) -> str:
    lines = [
        f"Linear-decodability audit  (F7 §1b)",
        f"dataset   : {report['dataset']}",
        f"windows   : {report['n_train']} train / {report['n_test']} test",
        f"input dim : {report['input_dim']}  (flattened history window)",
        f"threshold : R² > {report['threshold']} disqualifies a target as a headline result",
        "",
        f"{'target':<16}{'expected':<16}{'R²_raw':>8}   verdict",
        "-" * 60,
    ]
    surprises = []
    for name, r in report["targets"].items():
        verdict = "DISQUALIFIED (trivial)" if r["disqualified"] else "eligible"
        lines.append(f"{name:<16}{r['expected_kind']:<16}{r['r2_raw_window']:>8.4f}   {verdict}")
        expected_trivial = r["expected_kind"] == "linear_trivial"
        if expected_trivial != r["disqualified"]:
            surprises.append(name)

    lines.append("")
    if surprises:
        lines.append(
            "DISAGREEMENT between expected kind and measured decodability: "
            + ", ".join(surprises)
            + ". The measurement wins; update targets.py and say so."
        )
    else:
        lines.append("Measured decodability matches the expected taxonomy for every target.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="data/windows_v1", help="built dataset root")
    ap.add_argument("--alpha", type=float, default=1e-3, help="ridge regularisation")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--json-out", default=None, help="also write the raw report as JSON")
    args = ap.parse_args()

    report = audit(args.dataset, alpha=args.alpha, threshold=args.threshold)
    print(format_report(report))

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
