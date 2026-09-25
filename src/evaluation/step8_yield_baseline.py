"""Compare a yield-only defect classifier on the frozen Step 6 test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.step6_classify import LABELS, cross_split_keep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("data/processed/wafer_features.csv"))
    parser.add_argument("--map-cache", type=Path, default=Path("data/processed/step6_labeled_maps.npz"))
    parser.add_argument("--step6-metrics", type=Path, default=Path("reports/step6_metrics.json"))
    parser.add_argument("--report-output", type=Path, default=Path("reports"))
    args = parser.parse_args()
    args.report_output.mkdir(parents=True, exist_ok=True)
    figures = args.report_output / "figures"
    figures.mkdir(exist_ok=True)
    frame = pd.read_csv(args.features, usecols=["source_row", "lot_id", "failure_type", "split", "fail_rate"])
    frame = frame[frame.failure_type.isin(LABELS)].reset_index(drop=True)
    cache = np.load(args.map_cache)
    if not np.array_equal(cache["source_rows"], frame.source_row.to_numpy()):
        raise ValueError("Step 6 map cache does not match features")
    keep, removed = cross_split_keep(cache["hashes"], frame.split.to_numpy())
    frame = frame.loc[keep].reset_index(drop=True)
    train = frame.split == "train"
    test = frame.split == "test"
    if not np.isfinite(frame.fail_rate).all():
        raise ValueError("Yield-only input contains non-finite values")
    if set(frame.loc[train, "lot_id"]) & set(frame.loc[test, "lot_id"]):
        raise AssertionError("Lot leakage")
    model = Pipeline([("scaler", StandardScaler()),
                      ("classifier", LogisticRegression(class_weight="balanced", max_iter=500, random_state=42))])
    model.fit(frame.loc[train, ["fail_rate"]], frame.loc[train, "failure_type"])
    pred = model.predict(frame.loc[test, ["fail_rate"]])
    result = classification_report(frame.loc[test, "failure_type"], pred, labels=LABELS,
                                   output_dict=True, zero_division=0)
    summary = {"model": "yield_only_logistic", "features": ["fail_rate"],
               "train_count": int(train.sum()), "test_count": int(test.sum()),
               "duplicates_removed": removed,
               "macro_f1": result["macro avg"]["f1-score"],
               "accuracy": result["accuracy"], "per_class": result}
    (args.report_output / "step8_yield_baseline.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    prior = json.loads(args.step6_metrics.read_text(encoding="utf-8"))
    comparison = pd.DataFrame([
        {"model": "수율만", "macro_f1": summary["macro_f1"]},
        {"model": "공간 특징", "macro_f1": prior["logistic"]["macro_f1"]},
        {"model": "CNN map", "macro_f1": prior["CNN"]["macro_f1"]},
    ])
    comparison.to_csv(args.report_output / "step8_model_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(["Yield only", "Spatial features", "CNN wafer map"], comparison.macro_f1,
                  color=["#9aa5b1", "#277da1", "#43aa8b"])
    ax.bar_label(bars, fmt="%.3f", padding=4)
    ax.set(ylabel="Test macro-F1 (8 defect classes)", ylim=(0, 1), title="Same lot-split test set")
    fig.tight_layout()
    fig.savefig(figures / "step8_model_comparison.png", dpi=160)
    plt.close(fig)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
