"""Step 3: Moran's I and fail-fail join counts on valid wafer die."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.step1_yield import load_legacy_pickle, unwrap_label


FIELDS = [
    "source_row",
    "lot_id",
    "wafer_index",
    "failure_type",
    "yield",
    "valid_dies",
    "fail_dies",
    "neighbor_edges",
    "fail_fail_edges",
    "expected_fail_fail_edges",
    "moran_i",
]


def spatial_statistics(wafer_map: np.ndarray) -> dict[str, float | int]:
    """Binary Moran's I with undirected four-neighbor edges, outside die excluded."""
    array = np.asarray(wafer_map)
    if array.ndim != 2 or array.size == 0 or not np.isin(array, [0, 1, 2]).all():
        raise ValueError("Expected nonempty 2D wafer map with states 0, 1, 2")
    valid = array != 0
    fail = array == 2
    n = int(valid.sum())
    f = int(fail.sum())
    horizontal = valid[:, :-1] & valid[:, 1:]
    vertical = valid[:-1, :] & valid[1:, :]
    edges = int(horizontal.sum() + vertical.sum())
    ff_edges = int(
        (horizontal & fail[:, :-1] & fail[:, 1:]).sum()
        + (vertical & fail[:-1, :] & fail[1:, :]).sum()
    )
    expected_ff = (
        edges * f * (f - 1) / (n * (n - 1)) if n > 1 else float("nan")
    )
    if not (0 < f < n) or edges == 0:
        moran = float("nan")
    else:
        p = f / n
        centered = fail.astype(np.float32) - p
        pair_sum = float(
            np.sum((centered[:, :-1] * centered[:, 1:])[horizontal], dtype=np.float64)
            + np.sum((centered[:-1, :] * centered[1:, :])[vertical], dtype=np.float64)
        )
        denominator = n * p * (1 - p)
        moran = (n / edges) * pair_sum / denominator
    return {
        "valid_dies": n,
        "fail_dies": f,
        "neighbor_edges": edges,
        "fail_fail_edges": ff_edges,
        "expected_fail_fail_edges": expected_ff,
        "moran_i": moran,
    }


def permutation_test(
    wafer_map: np.ndarray, *, permutations: int = 999, seed: int = 2026
) -> tuple[float, float, float, np.ndarray]:
    """Shuffle fail labels only among measured die while keeping mask and fail count."""
    observed = spatial_statistics(wafer_map)
    value = observed["moran_i"]
    if np.isnan(value):
        return float("nan"), float("nan"), float("nan"), np.array([])
    array = np.asarray(wafer_map)
    valid = array != 0
    fail = array[valid] == 2
    shuffled = np.zeros_like(array, dtype=np.uint8)
    shuffled[valid] = 1
    rng = np.random.default_rng(seed)
    simulated = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        shuffled[valid] = np.where(rng.permutation(fail), 2, 1)
        simulated[index] = spatial_statistics(shuffled)["moran_i"]
    p_right = (1 + int((simulated >= value).sum())) / (permutations + 1)
    center = simulated.mean()
    p_two_sided = (
        1 + int((np.abs(simulated - center) >= abs(value - center)).sum())
    ) / (permutations + 1)
    z_score = (value - center) / simulated.std(ddof=1)
    return p_right, p_two_sided, z_score, simulated


def summarize_by_label(frame: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for label, group in frame.groupby("failure_type", sort=False):
        defined = group["moran_i"].dropna()
        summaries.append(
            {
                "failure_type": label,
                "wafers": len(group),
                "defined_moran": len(defined),
                "median_moran": defined.median() if len(defined) else np.nan,
                "q25_moran": defined.quantile(0.25) if len(defined) else np.nan,
                "q75_moran": defined.quantile(0.75) if len(defined) else np.nan,
                "positive_fraction": (defined > 0).mean() if len(defined) else np.nan,
            }
        )
    return pd.DataFrame(summaries).sort_values("wafers", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/LSWMD.pkl"))
    parser.add_argument("--data-output", type=Path, default=Path("data/processed"))
    parser.add_argument("--report-output", type=Path, default=Path("reports"))
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--examples-per-label", type=int, default=5)
    args = parser.parse_args()
    if args.permutations < 1 or args.examples_per_label < 1:
        raise ValueError("permutations and examples-per-label must be positive")
    args.data_output.mkdir(parents=True, exist_ok=True)
    figures = args.report_output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    frame = load_legacy_pickle(args.input)
    spatial_path = args.data_output / "wafer_spatial.csv"
    with spatial_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for source_row, row in enumerate(frame.itertuples(index=False)):
            stats = spatial_statistics(row.waferMap)
            valid = stats["valid_dies"]
            writer.writerow(
                {
                    "source_row": source_row,
                    "lot_id": row.lotName,
                    "wafer_index": int(row.waferIndex),
                    "failure_type": unwrap_label(row.failureType),
                    "yield": (valid - stats["fail_dies"]) / valid if valid else np.nan,
                    **stats,
                }
            )
            if (source_row + 1) % 100_000 == 0:
                print(f"Processed {source_row + 1:,} wafers", flush=True)

    result = pd.read_csv(spatial_path, encoding="utf-8-sig")
    if len(result) != len(frame) or not np.array_equal(
        result["source_row"].to_numpy(), np.arange(len(frame))
    ):
        raise ValueError("Spatial rows no longer align with source wafer maps")
    labels = summarize_by_label(result)
    labels.to_csv(args.report_output / "moran_by_label.csv", index=False)

    sample_rows: list[int] = []
    for _, group in result.groupby("failure_type", sort=False):
        defined = group.loc[group["moran_i"].notna()].sort_values("yield")
        if defined.empty:
            continue
        positions = np.linspace(
            0, len(defined) - 1, min(args.examples_per_label, len(defined)), dtype=int
        )
        sample_rows.extend(defined.iloc[positions]["source_row"].astype(int).tolist())
    case_index = args.report_output / "step2_case_index.csv"
    if case_index.exists():
        cases = pd.read_csv(case_index, encoding="utf-8-sig")
        low_lot = cases.loc[cases["selection"] == "lot wafer sequence"].sort_values(
            "yield"
        )
        if not low_lot.empty:
            sample_rows.append(int(low_lot.iloc[0]["source_row"]))
    sample_rows = sorted(set(sample_rows))

    permutation_rows = []
    null_example = None
    for position, source_row in enumerate(sample_rows):
        wafer = frame.iloc[source_row].waferMap
        p_right, p_two, z_score, simulated = permutation_test(
            wafer, permutations=args.permutations, seed=2026 + source_row
        )
        observed = result.iloc[source_row]
        permutation_rows.append(
            {
                "source_row": source_row,
                "lot_id": observed["lot_id"],
                "wafer_index": observed["wafer_index"],
                "failure_type": observed["failure_type"],
                "yield": observed["yield"],
                "moran_i": observed["moran_i"],
                "permutations": args.permutations,
                "p_right": p_right,
                "p_two_sided": p_two,
                "null_z_score": z_score,
            }
        )
        if null_example is None and observed["failure_type"] == "Center":
            null_example = (observed["moran_i"], simulated, source_row)
        if (position + 1) % 10 == 0:
            print(f"Permutation cases {position + 1}/{len(sample_rows)}", flush=True)
    permutation_frame = pd.DataFrame(permutation_rows)
    permutation_frame.to_csv(
        args.report_output / "moran_permutation_cases.csv", index=False
    )
    del frame

    plot_labels = labels.loc[labels["defined_moran"] > 0].iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(plot_labels))
    medians = plot_labels["median_moran"].to_numpy()
    ax.errorbar(
        medians,
        y,
        xerr=[
            medians - plot_labels["q25_moran"].to_numpy(),
            plot_labels["q75_moran"].to_numpy() - medians,
        ],
        fmt="o",
        color="#277da1",
        ecolor="#577590",
        capsize=3,
    )
    ax.axvline(0, color="#999999", linewidth=1)
    ax.set_yticks(y, plot_labels["failure_type"])
    ax.set(xlabel="Moran's I (median and interquartile range)", title="Spatial autocorrelation by label")
    fig.tight_layout()
    fig.savefig(figures / "step3_moran_by_label.png", dpi=160)
    plt.close(fig)

    if null_example is not None:
        observed_i, simulations, source_row = null_example
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(simulations, bins=35, color="#90be6d", edgecolor="white")
        ax.axvline(observed_i, color="#d1495b", linewidth=2, label=f"Observed I = {observed_i:.3f}")
        ax.set(
            xlabel="Moran's I under fixed fail-count random placement",
            ylabel="Permutations",
            title=f"Permutation reference: Center wafer, row {source_row}",
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / "step3_permutation_example.png", dpi=160)
        plt.close(fig)

    valid_i = result["moran_i"].dropna()
    positive_cases = int((permutation_frame["p_right"] <= 0.05).sum())
    report = [
        "# Step 3 · Moran's I와 공간 자기상관",
        "",
        "## 분석 범위와 계산",
        "",
        f"- 전체 **{len(result):,}장**에 global Moran's I와 fail-fail 이웃 쌍 수를 계산했다.",
        "- 원본 상태 0은 제외하고, 상태 1/2인 die만 4방향 상하좌우 인접 그래프의 노드로 사용했다.",
        "- fail을 1, pass를 0으로 두었다. Moran's I가 양수이면 비슷한 상태가 이웃할 경향이 있다는 뜻이다.",
        "- fail이 하나도 없거나 모든 유효 die가 fail인 wafer는 분산이 0이므로 Moran's I를 정의하지 않았다.",
        f"- {args.permutations:,}회 조건부 permutation은 전체가 아니라 라벨별 수율 범위에서 고른 **{len(permutation_frame)}장**에 적용했다. "
        "각 wafer의 mask와 fail die 수를 고정해 실제 배치와 비교했다.",
        "",
        "## 전체 결과",
        "",
        f"- Moran's I 정의 가능: **{len(valid_i):,} / {len(result):,}장**",
        f"- 전체 중앙값: **{valid_i.median():.3f}**; 25–75백분위: "
        f"**{valid_i.quantile(0.25):.3f}–{valid_i.quantile(0.75):.3f}**",
        "- fail-fail join count는 관찰된 이웃 fail 쌍이며, `expected_fail_fail_edges`는 동일한 fail die 수를 무작위로 놓았을 때의 기대값이다.",
        "",
        "| 라벨 | wafer 수 | 정의 가능 | Moran's I 중앙값 | 25–75백분위 |",
        "| :--- | ---: | ---: | ---: | ---: |",
        *[
            f"| {row.failure_type} | {row.wafers:,.0f} | {row.defined_moran:,.0f} "
            f"| {row.median_moran:.3f} | {row.q25_moran:.3f}–{row.q75_moran:.3f} |"
            for row in labels.itertuples()
        ],
        "",
        "![Moran by label](figures/step3_moran_by_label.png)",
        "",
        "## 조건부 무작위화 사례",
        "",
        f"- 사례 **{len(permutation_frame)}장** 중 **{positive_cases}장**이 양의 공간 자기상관에 대한 "
        "오른쪽 단측 permutation p-value ≤ 0.05였다.",
        "- 사례는 라벨과 수율을 기준으로 의도적으로 선정했으므로 이 비율을 전체 wafer의 유의 비율로 해석하지 않는다.",
        "- p-value는 사전에 정한 양의 자기상관 가설에 대한 값이다. 양·음 모두를 보려면 별도 기록한 `p_two_sided`를 사용한다.",
        "- [사례별 permutation 결과](moran_permutation_cases.csv)에 관찰값, p-value, 귀무분포 대비 z-score를 남겼다.",
        "",
        "![Permutation reference](figures/step3_permutation_example.png)",
        "",
        "## 검증과 해석 한계",
        "",
        "- 4×4 격자의 이진 checkerboard는 Moran's I = −1, 좌·우 반반 군집은 양수, 상수 map은 undefined로 확인했다.",
        "- 전체 계산 결과는 `data/processed/wafer_spatial.csv`에 보관하고 Git에는 올리지 않는다.",
        "- Moran's I는 공간적 유사성을 요약하지만 edge-ring, center, scratch를 단독으로 구별하지는 못한다. 다음 단계의 반경·방향·연결 성분 특징과 결합한다.",
        "- p-value는 같은 wafer 내 무작위 배치와의 차이를 나타낼 뿐, 특정 공정 원인을 증명하지 않는다.",
        "",
        "참고: [PySAL Moran's I 설명](https://pysal.org/esda/stable/user-guide/global_morans_i.html)",
    ]
    (args.report_output / "step3_moran.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(f"Report: {args.report_output / 'step3_moran.md'}")


if __name__ == "__main__":
    main()
