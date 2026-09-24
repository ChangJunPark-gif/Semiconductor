"""Step 4: wafer geometry features and train-only PCA."""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.step1_yield import load_legacy_pickle, unwrap_label


CONNECTIVITY = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
RADIAL_BOUNDS = (0.25, 0.50, 0.75)
PCA_FEATURES = [
    "edge_enrichment",
    "center_enrichment",
    "ring_0_enrichment",
    "ring_1_enrichment",
    "ring_2_enrichment",
    "ring_3_enrichment",
    "quadrant_asymmetry",
    "component_count_per_1000_valid",
    "largest_component_fail_share",
    "moran_i",
    "fail_join_excess_per_edge",
]
FIELDS = [
    "source_row",
    "lot_id",
    "wafer_index",
    "failure_type",
    "yield",
    "fail_rate",
    "valid_dies",
    "fail_dies",
    "edge_fail_rate",
    "center_fail_rate",
    "edge_enrichment",
    "center_enrichment",
    *[f"ring_{i}_fail_rate" for i in range(4)],
    *[f"ring_{i}_enrichment" for i in range(4)],
    *[f"quadrant_{i}_fail_rate" for i in range(4)],
    "quadrant_asymmetry",
    "component_count",
    "component_count_per_1000_valid",
    "largest_component_size",
    "largest_component_fail_share",
    "moran_i",
    "fail_join_excess_per_edge",
]


@lru_cache(maxsize=1024)
def geometry(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    yy, xx = np.indices(shape)
    cy, cx = (height - 1) / 2, (width - 1) / 2
    radius = np.sqrt(
        ((yy - cy) / max(cy, 1)) ** 2 + ((xx - cx) / max(cx, 1)) ** 2
    )
    quadrant = (yy >= cy).astype(np.uint8) * 2 + (xx >= cx).astype(np.uint8)
    return radius, quadrant


def region_rate(fail: np.ndarray, valid: np.ndarray, region: np.ndarray) -> float:
    members = valid & region
    denominator = int(members.sum())
    return float((fail & members).sum() / denominator) if denominator else float("nan")


def extract_features(
    wafer_map: np.ndarray, *, moran_i: float, fail_join_excess_per_edge: float
) -> dict[str, float | int]:
    array = np.asarray(wafer_map)
    if array.ndim != 2 or array.size == 0 or not np.isin(array, [0, 1, 2]).all():
        raise ValueError("Expected nonempty 2D wafer map with states 0, 1, 2")
    valid = array != 0
    fail = array == 2
    n = int(valid.sum())
    f = int(fail.sum())
    if n == 0:
        raise ValueError("No measured die")
    fail_rate = f / n
    radius, quadrant = geometry(array.shape)
    edge_rate = region_rate(fail, valid, radius >= 0.75)
    center_rate = region_rate(fail, valid, radius < 0.35)
    ring_masks = [
        radius < RADIAL_BOUNDS[0],
        (radius >= RADIAL_BOUNDS[0]) & (radius < RADIAL_BOUNDS[1]),
        (radius >= RADIAL_BOUNDS[1]) & (radius < RADIAL_BOUNDS[2]),
        radius >= RADIAL_BOUNDS[2],
    ]
    ring_rates = [region_rate(fail, valid, mask) for mask in ring_masks]
    quadrant_rates = [
        region_rate(fail, valid, quadrant == index) for index in range(4)
    ]
    finite_quadrants = np.asarray([value for value in quadrant_rates if np.isfinite(value)])
    asymmetry = (
        float(finite_quadrants.max() - finite_quadrants.min())
        if len(finite_quadrants) >= 2
        else float("nan")
    )
    if f:
        components, component_count = ndimage.label(fail, structure=CONNECTIVITY)
        sizes = np.bincount(components.ravel())[1:]
        largest_size = int(sizes.max())
    else:
        component_count = 0
        largest_size = 0

    features: dict[str, float | int] = {
        "yield": 1 - fail_rate,
        "fail_rate": fail_rate,
        "valid_dies": n,
        "fail_dies": f,
        "edge_fail_rate": edge_rate,
        "center_fail_rate": center_rate,
        "edge_enrichment": edge_rate - fail_rate,
        "center_enrichment": center_rate - fail_rate,
        "quadrant_asymmetry": asymmetry,
        "component_count": int(component_count),
        "component_count_per_1000_valid": 1000 * component_count / n,
        "largest_component_size": largest_size,
        "largest_component_fail_share": largest_size / f if f else 0,
        "moran_i": moran_i,
        "fail_join_excess_per_edge": fail_join_excess_per_edge,
    }
    for index, rate in enumerate(ring_rates):
        features[f"ring_{index}_fail_rate"] = rate
        features[f"ring_{index}_enrichment"] = rate - fail_rate
    for index, rate in enumerate(quadrant_rates):
        features[f"quadrant_{index}_fail_rate"] = rate
    return features


def assign_lot_split(lots: np.ndarray, *, seed: int = 42) -> dict[str, str]:
    """Assign whole lots to train/validation/test at 70/15/15 by lot count."""
    unique = np.unique(lots.astype(str))
    shuffled = np.random.default_rng(seed).permutation(unique)
    train_end = round(len(shuffled) * 0.70)
    val_end = round(len(shuffled) * 0.85)
    return {
        str(lot): ("train" if index < train_end else "validation" if index < val_end else "test")
        for index, lot in enumerate(shuffled)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/LSWMD.pkl"))
    parser.add_argument(
        "--spatial", type=Path, default=Path("data/processed/wafer_spatial.csv")
    )
    parser.add_argument("--data-output", type=Path, default=Path("data/processed"))
    parser.add_argument("--report-output", type=Path, default=Path("reports"))
    args = parser.parse_args()
    args.data_output.mkdir(parents=True, exist_ok=True)
    figures = args.report_output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    frame = load_legacy_pickle(args.input)
    features_path = args.data_output / "wafer_features.csv"
    with (
        args.spatial.open(newline="", encoding="utf-8-sig") as spatial_handle,
        features_path.open("w", newline="", encoding="utf-8-sig") as output_handle,
    ):
        spatial_reader = csv.DictReader(spatial_handle)
        writer = csv.DictWriter(output_handle, fieldnames=FIELDS)
        writer.writeheader()
        for source_row, wafer in enumerate(frame.itertuples(index=False)):
            spatial = next(spatial_reader, None)
            if spatial is None or int(spatial["source_row"]) != source_row:
                raise ValueError(f"Spatial results misaligned at row {source_row}")
            edges = int(spatial["neighbor_edges"])
            join_excess = (
                (float(spatial["fail_fail_edges"]) - float(spatial["expected_fail_fail_edges"]))
                / edges
                if edges
                else float("nan")
            )
            moran_value = float(spatial["moran_i"])
            features = extract_features(
                wafer.waferMap,
                moran_i=moran_value,
                fail_join_excess_per_edge=join_excess,
            )
            writer.writerow(
                {
                    "source_row": source_row,
                    "lot_id": wafer.lotName,
                    "wafer_index": int(wafer.waferIndex),
                    "failure_type": unwrap_label(wafer.failureType),
                    **features,
                }
            )
            if (source_row + 1) % 100_000 == 0:
                print(f"Extracted {source_row + 1:,} wafers", flush=True)
        if next(spatial_reader, None) is not None:
            raise ValueError("Spatial results have extra rows")
    del frame

    features = pd.read_csv(features_path, encoding="utf-8-sig")
    split_map = assign_lot_split(features["lot_id"].to_numpy())
    features["split"] = features["lot_id"].map(split_map)
    if features["split"].isna().any():
        raise ValueError("Some lots were not assigned a split")
    lot_split_path = args.data_output / "lot_split.csv"
    pd.DataFrame(
        [{"lot_id": lot, "split": split_name} for lot, split_name in split_map.items()]
    ).to_csv(lot_split_path, index=False)
    features.to_csv(features_path, index=False)

    train_mask = features["split"] == "train"
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, svd_solver="full", random_state=42)),
        ]
    )
    pipeline.fit(features.loc[train_mask, PCA_FEATURES])
    components = pipeline.transform(features[PCA_FEATURES]).astype(np.float32)
    np.save(args.data_output / "pca_embedding.npy", components)
    joblib.dump(pipeline, args.data_output / "pca_pipeline.joblib")

    imputed_names = list(pipeline.named_steps["imputer"].get_feature_names_out(PCA_FEATURES))
    pca = pipeline.named_steps["pca"]
    loadings = pd.DataFrame(
        pca.components_.T,
        index=imputed_names,
        columns=[f"PC{index + 1}" for index in range(pca.n_components_)],
    )
    loadings.to_csv(args.report_output / "pca_loadings.csv")
    variance = pd.DataFrame(
        {
            "component": np.arange(1, pca.n_components_ + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    variance.to_csv(args.report_output / "pca_variance.csv", index=False)

    split_counts = features.groupby("split").agg(
        wafers=("source_row", "size"), lots=("lot_id", "nunique")
    )
    label_split = pd.crosstab(features["failure_type"], features["split"])
    label_split.to_csv(args.report_output / "label_split_counts.csv")
    train_lots = set(features.loc[train_mask, "lot_id"])
    val_lots = set(features.loc[features.split == "validation", "lot_id"])
    test_lots = set(features.loc[features.split == "test", "lot_id"])
    if train_lots & val_lots or train_lots & test_lots or val_lots & test_lots:
        raise AssertionError("Lot leakage between splits")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(variance["component"], variance["cumulative_variance"], marker="o", color="#277da1")
    ax.axhline(0.95, linestyle="--", color="#f3722c")
    ax.set(
        xlabel="Principal components",
        ylabel="Cumulative explained variance",
        title="Train-fitted spatial PCA",
        ylim=(0, 1.02),
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "step4_pca_variance.png", dpi=160)
    plt.close(fig)

    if components.shape[1] >= 2:
        rng = np.random.default_rng(42)
        plot_indices = []
        for _, group in features.groupby("failure_type", sort=False):
            indices = group.index.to_numpy()
            plot_indices.extend(rng.choice(indices, size=min(500, len(indices)), replace=False))
        plot_indices = np.array(plot_indices, dtype=int)
        fig, ax = plt.subplots(figsize=(8, 6))
        for label in sorted(features["failure_type"].unique()):
            selected = plot_indices[features.iloc[plot_indices]["failure_type"].to_numpy() == label]
            ax.scatter(
                components[selected, 0],
                components[selected, 1],
                s=9,
                alpha=0.45,
                label=label,
            )
        ax.set(xlabel="PC1", ylabel="PC2", title="Spatial PCA: stratified label sample")
        ax.legend(markerscale=2, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / "step4_pca_pattern_scatter.png", dpi=160)
        plt.close(fig)

    correlation = np.corrcoef(components[:, 0], features["yield"].to_numpy())[0, 1]
    top_loadings = loadings["PC1"].abs().sort_values(ascending=False).head(5)
    report = [
        "# Step 4 · 공간 특징 추출과 PCA",
        "",
        "## 특징 정의",
        "",
        "변수별 계산식과 PCA 사용 여부는 [특징 사전](feature_dictionary.md)을 참조한다.",
        "",
        f"- 전체 **{len(features):,}장**에서 수율, 중심·가장자리·반경별 fail rate, 사분면 비대칭, 연결 성분, Moran's I 및 fail-fail 이웃 초과량을 계산했다.",
        "- 반경은 map 중심에서 세로·가로 반지름으로 정규화했다. 중심은 반경 < 0.35, 가장자리는 반경 ≥ 0.75다.",
        "- 각 영역의 fail rate는 해당 영역의 유효 die를 분모로 쓴다. enrichment는 영역 fail rate에서 wafer 전체 fail rate를 뺀 값이다.",
        "- 사분면 비대칭은 네 사분면 fail rate의 최대−최소다. 연결 성분은 fail die의 4방향 인접을 사용한다.",
        "- `moran_i`가 정의되지 않는 상수 map은 결측으로 보존한 뒤 train 중앙값으로 대체하고, PCA 입력에 결측 지시자를 추가한다.",
        "- permutation null z-score는 3단계에서 선정한 사례에만 있어 전체 PCA 입력에는 넣지 않았다. 대신 Moran's I와 관찰−기대 fail-fail edge 비율을 넣었다.",
        "",
        "## 누출 방지와 split",
        "",
        "- 고유 lot을 시드 42로 섞어 **lot 수 기준 70/15/15**로 train/validation/test에 배정했다. 세 세트의 lot 교집합은 0이다.",
        "- 원본 `trianTestLabel`은 lot 단위 분할이 보장되지 않아 모델링 split에 사용하지 않았다.",
        "- 결측 중앙값, 표준화 평균·분산, PCA 축은 **train wafer만** 사용해 적합했다. validation/test는 같은 파이프라인으로 변환만 했다.",
        "- 라벨 `failure_type`과 수율 `yield`은 분석 표에 보존하지만 PCA 입력에는 넣지 않았다. 공간 enrichment와 연결성·Moran 지표만 PCA 입력에 사용했다.",
        "",
        "| Split | Lot 수 | Wafer 수 |",
        "| :--- | ---: | ---: |",
        *[
            f"| {name} | {split_counts.loc[name, 'lots']:,} | {split_counts.loc[name, 'wafers']:,} |"
            for name in ("train", "validation", "test")
        ],
        "",
        "## PCA 결과",
        "",
        f"- 공간 특징 **{len(PCA_FEATURES)}개**와 결측 지시자를 표준화했다.",
        f"- train 설명 분산 95%에 도달하는 주성분 수: **{pca.n_components_}개**.",
        f"- PC1 설명 분산: **{pca.explained_variance_ratio_[0]:.1%}**.",
        f"- PC1과 wafer yield의 전체 Pearson 상관: **{correlation:.3f}**. 수율 자체를 PCA 입력에서 제외했어도 공간 특징과 수율이 연관될 수 있다.",
        f"- PC1 절대 적재량 상위 5개: **{', '.join(top_loadings.index)}**.",
        "",
        "![PCA explained variance](figures/step4_pca_variance.png)",
        "",
        "![PCA labeled sample](figures/step4_pca_pattern_scatter.png)",
        "",
        "## 생성 파일과 다음 단계",
        "",
        "- `data/processed/wafer_features.csv`: 전체 특징·라벨·lot split",
        "- `data/processed/lot_split.csv`: lot별 고정 split",
        "- `data/processed/pca_embedding.npy`, `pca_pipeline.joblib`: 후속 군집용 임베딩과 train에 적합한 변환기",
        "- [PCA 적재량](pca_loadings.csv), [설명 분산](pca_variance.csv), [라벨별 split 수](label_split_counts.csv)",
        "",
        "> [!NOTE]",
        "> PCA는 패턴을 분리해 준다는 보장이 없다. 군집 단계에서는 수율만으로 무리가 나뉘는지와 라벨별 대표 map이 일관되는지 따로 검토한다.",
    ]
    (args.report_output / "step4_features_pca.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(f"PCA components: {pca.n_components_}; report: {args.report_output / 'step4_features_pca.md'}")


if __name__ == "__main__":
    main()
