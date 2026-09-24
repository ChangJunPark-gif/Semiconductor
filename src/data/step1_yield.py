"""Step 1: validate WM-811K and compute wafer/lot yield and bin distributions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.core.indexes.base as legacy_index_base


SOURCE_URL = "https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map"
EXPECTED_COLUMNS = {
    "waferMap",
    "dieSize",
    "lotName",
    "waferIndex",
    "trianTestLabel",
    "failureType",
}
WAFER_FIELDS = [
    "source_row",
    "lot_id",
    "wafer_index",
    "height",
    "width",
    "outside_dies",
    "pass_dies",
    "fail_dies",
    "valid_dies",
    "yield",
    "fail_rate",
    "failure_type",
    "original_split",
    "die_size_metadata",
    "die_size_matches",
]


def load_legacy_pickle(path: Path) -> pd.DataFrame:
    """WM-811K was pickled with old pandas module paths and Python 2 strings."""
    sys.modules["pandas.indexes"] = pd.core.indexes
    sys.modules["pandas.indexes.base"] = legacy_index_base
    with path.open("rb") as handle:
        frame = pickle.load(handle, encoding="latin1")
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(frame).__name__}")
    missing = EXPECTED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return frame


def unwrap_label(value: object) -> str:
    while isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return "unlabeled"
        value = value[0]
    if value is None or pd.isna(value):
        return "unlabeled"
    return str(value)


def count_dies(wafer_map: object) -> tuple[int, int, int, int, int]:
    array = np.asarray(wafer_map)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("wafer map is not a nonempty 2D array")
    counts = np.bincount(array.ravel().astype(np.int64), minlength=3)
    if len(counts) != 3 or int(counts.sum()) != array.size:
        raise ValueError("wafer map has die states outside 0, 1, 2")
    return array.shape[0], array.shape[1], *(int(x) for x in counts)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/LSWMD.pkl"))
    parser.add_argument("--data-output", type=Path, default=Path("data/processed"))
    parser.add_argument("--report-output", type=Path, default=Path("reports"))
    args = parser.parse_args()

    args.data_output.mkdir(parents=True, exist_ok=True)
    figures = args.report_output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    frame = load_legacy_pickle(args.input)
    lot_stats: dict[str, dict] = defaultdict(
        lambda: {"wafers": 0, "pass": 0, "fail": 0, "yield_sum": 0.0}
    )
    label_stats: dict[str, dict] = defaultdict(
        lambda: {"wafers": 0, "pass": 0, "fail": 0, "yield_sum": 0.0}
    )
    original_splits = Counter()
    shape_counts = Counter()
    invalid_reasons = Counter()
    missing_lot_count = 0
    missing_index_count = 0
    metadata_mismatch_count = 0
    seen_wafer_keys = set()
    duplicate_wafer_key_count = 0
    totals = np.zeros(3, dtype=np.int64)
    yields: list[float] = []
    n_valid_wafers = 0
    wafer_csv = args.data_output / "wafer_yield.csv"

    with wafer_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=WAFER_FIELDS)
        writer.writeheader()
        for source_row, row in enumerate(frame.itertuples(index=False)):
            try:
                height, width, outside, passed, failed = count_dies(row.waferMap)
            except (TypeError, ValueError) as error:
                invalid_reasons[str(error)] += 1
                continue
            valid = passed + failed
            if valid == 0:
                invalid_reasons["no measured pass/fail dies"] += 1
                continue

            lot_id = str(row.lotName) if pd.notna(row.lotName) else ""
            wafer_index = int(row.waferIndex) if pd.notna(row.waferIndex) else ""
            missing_lot_count += int(not lot_id)
            missing_index_count += int(wafer_index == "")
            if lot_id and wafer_index != "":
                key = (lot_id, wafer_index)
                duplicate_wafer_key_count += int(key in seen_wafer_keys)
                seen_wafer_keys.add(key)

            label = unwrap_label(row.failureType)
            split = unwrap_label(row.trianTestLabel)
            original_splits[split] += 1
            shape_counts[(height, width)] += 1
            yield_value = passed / valid
            yields.append(yield_value)
            totals += (outside, passed, failed)
            n_valid_wafers += 1

            size_meta = float(row.dieSize) if pd.notna(row.dieSize) else None
            size_matches = size_meta is not None and int(size_meta) == valid
            metadata_mismatch_count += int(not size_matches)

            for stats in (lot_stats[lot_id], label_stats[label]):
                stats["wafers"] += 1
                stats["pass"] += passed
                stats["fail"] += failed
                stats["yield_sum"] += yield_value

            writer.writerow(
                {
                    "source_row": source_row,
                    "lot_id": lot_id,
                    "wafer_index": wafer_index,
                    "height": height,
                    "width": width,
                    "outside_dies": outside,
                    "pass_dies": passed,
                    "fail_dies": failed,
                    "valid_dies": valid,
                    "yield": f"{yield_value:.8f}",
                    "fail_rate": f"{1-yield_value:.8f}",
                    "failure_type": label,
                    "original_split": split,
                    "die_size_metadata": size_meta,
                    "die_size_matches": size_matches,
                }
            )
            if n_valid_wafers % 100_000 == 0:
                print(f"Processed {n_valid_wafers:,} wafers", flush=True)

    del frame
    labels = [
        {
            "failure_type": label,
            "wafers": stats["wafers"],
            "mean_wafer_yield": stats["yield_sum"] / stats["wafers"],
            "pooled_die_yield": stats["pass"] / (stats["pass"] + stats["fail"]),
        }
        for label, stats in label_stats.items()
    ]
    labels.sort(key=lambda row: row["wafers"], reverse=True)
    write_csv(
        args.report_output / "label_yield.csv",
        ["failure_type", "wafers", "mean_wafer_yield", "pooled_die_yield"],
        labels,
    )
    lots = [
        {
            "lot_id": lot,
            "wafers": stats["wafers"],
            "mean_wafer_yield": stats["yield_sum"] / stats["wafers"],
            "pooled_die_yield": stats["pass"] / (stats["pass"] + stats["fail"]),
        }
        for lot, stats in lot_stats.items()
    ]
    lots.sort(key=lambda row: row["lot_id"])
    write_csv(
        args.data_output / "lot_yield.csv",
        ["lot_id", "wafers", "mean_wafer_yield", "pooled_die_yield"],
        lots,
    )

    values = np.asarray(yields)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(values, bins=np.linspace(0, 1, 51), color="#277da1", edgecolor="white")
    ax.set(xlabel="Wafer yield (pass / measured dies)", ylabel="Number of wafers")
    ax.set_title("WM-811K wafer yield distribution")
    fig.tight_layout()
    fig.savefig(figures / "wafer_yield_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_labels = sorted(labels, key=lambda item: item["wafers"])
    ax.barh(
        [item["failure_type"] for item in sorted_labels],
        [item["wafers"] for item in sorted_labels],
        color="#577590",
    )
    ax.set_xscale("log")
    ax.set(xlabel="Number of wafers (log scale)", title="WM-811K pattern labels")
    fig.tight_layout()
    fig.savefig(figures / "pattern_label_distribution.png", dpi=160)
    plt.close(fig)

    q = np.quantile(values, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
    total_die_count = int(totals.sum())
    measured_die_count = int(totals[1] + totals[2])
    report = [
        "# Step 1 · WM-811K 데이터 품질과 수율·bin 분포",
        "",
        f"- 데이터 출처: [WM-811K]({SOURCE_URL})",
        f"- 원본 파일: `{args.input.as_posix()}` ({args.input.stat().st_size:,} bytes)",
        f"- SHA-256: `{sha256(args.input)}`",
        f"- 원본 행: **{n_valid_wafers + sum(invalid_reasons.values()):,}**",
        f"- 분석 가능한 wafer: **{n_valid_wafers:,}**",
        f"- 고유 lot: **{len(lot_stats):,}**",
        "",
        "## 상태 코드와 계산식",
        "",
        "| 원본 코드 | 의미 | 수율 분모 포함 |",
        "| :--- | :--- | :--- |",
        "| 0 | wafer 바깥 또는 분석에서 제외한 die | 아니오 |",
        "| 1 | pass | 예 |",
        "| 2 | fail | 예 |",
        "",
        "`wafer yield = pass_dies / (pass_dies + fail_dies)`. 유효 die가 없는 map은 분석에서 제외했다. 원본의 `dieSize`와 유효 die 개수를 비교했다.",
        "",
        "## 전체 bin 분포",
        "",
        "| Bin | Die 수 | 전체 격자 대비 |",
        "| :--- | ---: | ---: |",
        *[
            f"| {name} ({code}) | {int(totals[code]):,} | {totals[code]/total_die_count:.2%} |"
            for code, name in enumerate(("outside", "pass", "fail"))
        ],
        "",
        f"측정된 die 중 pass 비율은 **{totals[1]/measured_die_count:.2%}**다. 이는 wafer별 yield의 평균과 다른, die 수로 가중된 전체 비율이다.",
        "",
        "## Wafer yield 분포",
        "",
        f"- 평균 wafer yield: **{values.mean():.2%}**",
        f"- 중앙값: **{q[3]:.2%}**",
        f"- 최소 / 5% / 25% / 50% / 75% / 95% / 최대: **{' / '.join(f'{x:.2%}' for x in q)}**",
        "",
        "![Wafer yield distribution](figures/wafer_yield_distribution.png)",
        "",
        "## 라벨 분포와 yield",
        "",
        "| 패턴 | wafer 수 | 평균 wafer yield |",
        "| :--- | ---: | ---: |",
        *[
            f"| {item['failure_type']} | {item['wafers']:,} | {item['mean_wafer_yield']:.2%} |"
            for item in labels
        ],
        "",
        "![Pattern label distribution](figures/pattern_label_distribution.png)",
        "",
        "## 품질 점검",
        "",
        f"- 제외 wafer: **{sum(invalid_reasons.values()):,}** ({dict(invalid_reasons)})",
        f"- `dieSize`와 유효 die 수 불일치/결측: **{metadata_mismatch_count:,}**",
        f"- lot ID 결측: **{missing_lot_count:,}**",
        f"- waferIndex 결측: **{missing_index_count:,}**",
        f"- 중복 `(lot_id, wafer_index)` 키: **{duplicate_wafer_key_count:,}** (원본 행 번호로 구별)",
        f"- 원본 `trianTestLabel` 분포: **{dict(original_splits)}**",
        f"- 가장 많은 map 크기 5종: **{shape_counts.most_common(5)}**",
        "",
        "## 생성 파일",
        "",
        "- `data/processed/wafer_yield.csv`: wafer별 die 수·수율·라벨",
        "- `data/processed/lot_yield.csv`: lot별 평균 wafer yield와 die 가중 수율",
        "- `reports/label_yield.csv`: 라벨별 샘플 수와 수율",
        "- `reports/figures/`: 보고서 그림",
        "",
        "> [!NOTE]",
        "> 공개 데이터의 패턴 라벨은 일부 wafer에만 있다. lot·라벨별 수율 차이는 공정 원인을 뜻하지 않으며, 다음 단계의 공간 분석을 위한 탐색 결과다.",
    ]
    (args.report_output / "data_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(f"Report: {args.report_output / 'data_report.md'}")


if __name__ == "__main__":
    main()
