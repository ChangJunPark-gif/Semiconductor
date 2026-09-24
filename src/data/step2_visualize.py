"""Step 2: render original-grid WM-811K wafer maps and lot sequences."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from src.data.step1_yield import load_legacy_pickle


PATTERN_ORDER = [
    "none",
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Random",
    "Scratch",
    "Near-full",
    "unlabeled",
]
DEFECT_PATTERNS = [name for name in PATTERN_ORDER if name not in {"none", "unlabeled"}]
COLORS = ["#edf1f4", "#8bc6a0", "#cf4e49"]
CMAP = ListedColormap(COLORS)
NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], CMAP.N)
LEGEND = [
    Patch(facecolor=color, edgecolor="#54606d", label=label)
    for color, label in zip(COLORS, ("outside (0)", "pass (1)", "fail (2)"))
]


def choose_nearest(group: pd.DataFrame, target: float) -> pd.Series:
    position = (group["yield"] - target).abs().to_numpy().argmin()
    return group.iloc[int(position)]


def draw_gallery(
    frame: pd.DataFrame,
    cases: list[dict],
    path: Path,
    title: str,
    columns: int = 5,
) -> None:
    rows = int(np.ceil(len(cases) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3.5, rows * 3.5))
    axes = np.asarray(axes).ravel()
    for ax, case in zip(axes, cases):
        wafer_map = np.asarray(frame.iloc[case["source_row"]].waferMap)
        if wafer_map.ndim != 2 or not np.isin(wafer_map, [0, 1, 2]).all():
            raise ValueError(f"Unexpected map states at row {case['source_row']}")
        ax.imshow(wafer_map, cmap=CMAP, norm=NORM, interpolation="nearest", aspect="equal")
        ax.set_title(
            f"{case['failure_type']} · {case['yield']:.1%}\n"
            f"{case['lot_id']} / wafer {case['wafer_index']}",
            fontsize=9,
        )
        ax.set_axis_off()
    for ax in axes[len(cases) :]:
        ax.set_axis_off()
    fig.suptitle(title, fontsize=15)
    fig.legend(handles=LEGEND, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/LSWMD.pkl"))
    parser.add_argument(
        "--summary", type=Path, default=Path("data/processed/wafer_yield.csv")
    )
    parser.add_argument("--report-output", type=Path, default=Path("reports"))
    args = parser.parse_args()

    figures = args.report_output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.summary, encoding="utf-8-sig")
    required = {"source_row", "lot_id", "wafer_index", "yield", "failure_type"}
    if missing := required.difference(summary.columns):
        raise ValueError(f"Missing summary columns: {sorted(missing)}")

    representative: list[dict] = []
    lower_tail: list[dict] = []
    for pattern in PATTERN_ORDER:
        group = summary.loc[summary.failure_type == pattern]
        if group.empty:
            raise ValueError(f"No wafer rows for pattern {pattern}")
        median_case = choose_nearest(group, float(group["yield"].median()))
        representative.append(median_case.to_dict() | {"selection": "class median yield"})
        if pattern in DEFECT_PATTERNS:
            low_case = choose_nearest(group, float(group["yield"].quantile(0.05)))
            lower_tail.append(low_case.to_dict() | {"selection": "class 5th-percentile yield"})

    lot_summary = summary.groupby("lot_id", sort=False)["yield"].agg(
        ["count", "min", "max"]
    )
    eligible = lot_summary.loc[lot_summary["count"] >= 15].copy()
    eligible["spread"] = eligible["max"] - eligible["min"]
    if eligible.empty:
        raise ValueError("No lot has at least 15 wafers")
    chosen_lot = str(eligible.sort_values("spread", ascending=False).index[0])
    lot_rows = summary.loc[summary.lot_id == chosen_lot].sort_values("wafer_index")
    lowest_wafer = lot_rows.loc[lot_rows["yield"].idxmin()]
    lot_yields = lot_rows["yield"].to_numpy()
    positions = sorted(
        set(np.linspace(0, len(lot_rows) - 1, min(10, len(lot_rows)), dtype=int))
        | {int(lot_yields.argmin()), int(lot_yields.argmax())}
    )
    lot_cases = [
        row.to_dict() | {"selection": "lot wafer sequence"}
        for _, row in lot_rows.iloc[positions].iterrows()
    ]

    selected = representative + lower_tail + lot_cases
    frame = load_legacy_pickle(args.input)
    if len(frame) != 811_457:
        raise ValueError(f"Unexpected source length: {len(frame)}")
    draw_gallery(
        frame,
        representative,
        figures / "step2_pattern_gallery.png",
        "Representative maps by pattern label",
        columns=5,
    )
    draw_gallery(
        frame,
        lower_tail,
        figures / "step2_low_yield_gallery.png",
        "Lower-yield example within each defect label",
        columns=4,
    )
    draw_gallery(
        frame,
        lot_cases,
        figures / "step2_lot_sequence.png",
        f"Wafer sequence in {chosen_lot}",
        columns=4,
    )

    case_path = args.report_output / "step2_case_index.csv"
    with case_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = [
            "selection",
            "source_row",
            "lot_id",
            "wafer_index",
            "failure_type",
            "yield",
            "height",
            "width",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: case[key] for key in fields} for case in selected])

    report = [
        "# Step 2 · wafer 공간 시각화",
        "",
        "원본 격자를 유지한 wafer map으로 상태 코드 0/1/2를 서로 다른 색으로 표시했다. "
        "그림은 분석용 PNG이며, 분류 모델 입력을 위한 리사이즈는 아직 수행하지 않았다.",
        "",
        "## 색상 및 해석",
        "",
        "| 값 | 색상 | 의미 |",
        "| :--- | :--- | :--- |",
        "| 0 | 연회색 | wafer 바깥 |",
        "| 1 | 녹색 | pass die |",
        "| 2 | 적색 | fail die |",
        "",
        "각 그림은 `interpolation='nearest'`를 사용해 die를 흐리게 보간하지 않는다. "
        "가로·세로 비율은 원본 map을 따른다. 바깥 영역은 fail로 칠하지 않는다.",
        "",
        "## 패턴별 대표 map",
        "",
        "각 라벨의 wafer yield 중앙값에 가장 가까운 실제 wafer 한 장씩을 골랐다. "
        "여기에는 패턴 없음(`none`)과 미라벨(`unlabeled`)이 포함된다.",
        "",
        "![Pattern gallery](figures/step2_pattern_gallery.png)",
        "",
        "## 낮은 수율 사례",
        "",
        "8개 불량 라벨에서 각각 수율 5백분위수에 가까운 wafer를 골랐다. "
        "같은 라벨 내부에서도 fail die의 양과 형태가 달라지는지 비교한다.",
        "",
        "![Low yield gallery](figures/step2_low_yield_gallery.png)",
        "",
        "## lot 내 wafer 순서",
        "",
        f"15장 이상인 lot 가운데 수율 범위가 가장 큰 **{chosen_lot}**을 선택했다. "
        f"이 lot의 총 wafer 수는 **{len(lot_rows)}장**, 최저·최고 수율은 "
        f"**{lot_rows['yield'].min():.1%} / {lot_rows['yield'].max():.1%}**다. "
        "wafer index 순서에 따라 최대 12장을 추출하되 최저·최고 수율 wafer를 반드시 포함했다.",
        "",
        f"특히 wafer **{int(lowest_wafer['wafer_index'])}**의 수율은 "
        f"**{lowest_wafer['yield']:.1%}**로 같은 lot의 다른 wafer와 크게 다르다. "
        "이는 후속 조사 우선순위를 정할 사례이며, 원인은 공정 이력 없이 특정할 수 없다.",
        "",
        "![Lot wafer sequence](figures/step2_lot_sequence.png)",
        "",
        "## 재현성과 품질 점검",
        "",
        f"- 원본 811,457장 중 대표 10장, 라벨별 저수율 8장, lot 순서 {len(lot_cases)}장을 선택했다.",
        "- 선정 근거와 원본 행 번호는 [사례 인덱스](step2_case_index.csv)에 기록했다.",
        "- 모든 선택은 1단계 wafer별 CSV로 결정하고, 그림은 해당 원본 `waferMap`에서 직접 만들었다.",
        "- 그림 작성 전에 각 map이 2차원이며 상태값이 0/1/2뿐인지 다시 검증했다.",
        "",
        "> [!NOTE]",
        "> lot 내 모양의 유사성은 원인에 대한 가설을 만드는 단서일 뿐이다. "
        "공정 이력 없이 장비 또는 공정 단계를 특정하지 않는다.",
    ]
    report_path = args.report_output / "step2_visualization.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Selected lot: {chosen_lot}; report: {report_path}")


if __name__ == "__main__":
    main()
