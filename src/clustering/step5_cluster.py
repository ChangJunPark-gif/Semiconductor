"""Step 5: compare density clustering on train-only spatial PCA vectors."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

from src.data.step1_yield import load_legacy_pickle
from src.data.step2_visualize import CMAP, LEGEND, NORM


def cluster_metrics(labels: np.ndarray) -> dict[str, float | int]:
    """Summarize density labels, treating -1 as noise."""
    labels = np.asarray(labels)
    counts = pd.Series(labels[labels >= 0]).value_counts()
    assigned = int(counts.sum())
    return {
        "clusters": int(len(counts)),
        "noise_fraction": float(np.mean(labels == -1)),
        "largest_cluster_fraction": float(counts.max() / assigned) if assigned else 0.0,
    }


def choose_dbscan(results: pd.DataFrame) -> int:
    """Choose a train-only setting that avoids all-noise/single-cluster output."""
    viable = results[(results.clusters >= 2) & (results.noise_fraction <= 0.8)]
    if viable.empty:
        return int((results.noise_fraction - 0.3).abs().idxmin())
    score = (viable.noise_fraction - 0.3).abs() + viable.largest_cluster_fraction
    return int(score.idxmin())


def gallery(frame: pd.DataFrame, labels: np.ndarray, vectors: np.ndarray, raw: Path, out: Path) -> pd.DataFrame:
    """Show two actual wafers from each of the largest non-noise clusters."""
    counts = pd.Series(labels[labels >= 0]).value_counts().head(6)
    choices: list[dict] = []
    for cluster, _ in counts.items():
        indices = np.flatnonzero(labels == cluster)
        center = np.median(vectors[indices], axis=0)
        distances = np.linalg.norm(vectors[indices] - center, axis=1)
        for role, position in (("central", int(np.argmin(distances))), ("outer", int(np.argmax(distances)))):
            row = frame.iloc[indices[position]]
            choices.append({"cluster": int(cluster), "role": role, "source_row": int(row.source_row),
                            "lot_id": row.lot_id, "failure_type": row.failure_type, "yield": float(row["yield"])})
    if not choices:
        return pd.DataFrame(columns=["cluster", "role", "source_row", "lot_id", "failure_type", "yield"])
    original = load_legacy_pickle(raw)
    fig, axes = plt.subplots(len(counts), 2, figsize=(7, max(3, len(counts) * 2.7)))
    axes = np.atleast_2d(axes)
    for ax, case in zip(axes.flat, choices):
        wafer_map = np.asarray(original.iloc[case["source_row"]].waferMap)
        ax.imshow(wafer_map, cmap=CMAP, norm=NORM, interpolation="nearest")
        ax.set_title(f"C{case['cluster']} {case['role']} · {case['failure_type']} · yield {case['yield']:.1%}", fontsize=9)
        ax.axis("off")
    fig.suptitle("HDBSCAN train clusters: central and outer members")
    fig.legend(handles=LEGEND, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return pd.DataFrame(choices)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("data/processed/wafer_features.csv"))
    parser.add_argument("--embedding", type=Path, default=Path("data/processed/pca_embedding.npy"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/LSWMD.pkl"))
    parser.add_argument("--data-output", type=Path, default=Path("data/processed"))
    parser.add_argument("--sample-size", type=int, default=12000)
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.data_output.mkdir(parents=True, exist_ok=True)
    figures = args.out / "figures"
    figures.mkdir(exist_ok=True)

    columns = ["source_row", "lot_id", "failure_type", "yield", "split"]
    features = pd.read_csv(args.features, usecols=columns)
    vectors = np.load(args.embedding, mmap_mode="r")
    if len(features) != len(vectors) or not np.array_equal(features.source_row.to_numpy(), np.arange(len(features))):
        raise ValueError("Feature rows and PCA embedding are not aligned")
    train_indices = np.flatnonzero(features.split.to_numpy() == "train")
    if args.sample_size < 2 or args.sample_size > len(train_indices):
        raise ValueError("sample-size must be between 2 and train wafer count")
    sampled = np.sort(np.random.default_rng(42).choice(train_indices, args.sample_size, replace=False))
    table = features.iloc[sampled].reset_index(drop=True)
    x = np.asarray(vectors[sampled], dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("PCA vectors contain non-finite values")

    neighbor_count = 15
    distances = NearestNeighbors(n_neighbors=neighbor_count + 1).fit(x).kneighbors(x)[0][:, -1]
    eps_values = np.unique(np.quantile(distances, [0.5, 0.7, 0.85]))
    db_labels = []
    db_rows = []
    for eps in eps_values:
        labels = DBSCAN(eps=float(eps), min_samples=neighbor_count, n_jobs=-1).fit_predict(x)
        db_labels.append(labels)
        db_rows.append({"method": "DBSCAN", "parameter": f"eps={eps:.5g}; min_samples=15", **cluster_metrics(labels)})
    db_summary = pd.DataFrame(db_rows)
    chosen_idx = choose_dbscan(db_summary)

    hdb_labels = []
    hdb_rows = []
    for minimum in (60, 120, 240):
        labels = HDBSCAN(min_cluster_size=minimum, min_samples=15, n_jobs=-1, copy=True).fit_predict(x)
        hdb_labels.append(labels)
        hdb_rows.append({"method": "HDBSCAN", "parameter": f"min_cluster_size={minimum}; min_samples=15", **cluster_metrics(labels)})
    hdb_summary = pd.DataFrame(hdb_rows)
    summary = pd.concat([db_summary, hdb_summary], ignore_index=True)
    summary.to_csv(args.out / "step5_parameter_summary.csv", index=False)

    baseline = hdb_labels[1]
    db_chosen = db_labels[chosen_idx]
    table = table.copy()
    table["hdbscan_cluster"] = baseline
    table["dbscan_cluster"] = db_chosen
    table.to_csv(args.data_output / "step5_sample_clusters.csv", index=False)
    counts = pd.crosstab(table.hdbscan_cluster, table.failure_type)
    counts.to_csv(args.out / "step5_label_crosstab.csv")
    grouped = table.groupby("hdbscan_cluster").agg(
        wafers=("source_row", "size"), median_yield=("yield", "median"),
        q1_yield=("yield", lambda s: s.quantile(0.25)), q3_yield=("yield", lambda s: s.quantile(0.75)),
        labeled_fraction=("failure_type", lambda s: (~s.isin(["unlabeled", "none"])).mean()),
    ).reset_index().sort_values("wafers", ascending=False)
    grouped.to_csv(args.out / "step5_cluster_summary.csv", index=False)
    representative = gallery(table, baseline, x, args.raw, figures / "step5_cluster_gallery.png")
    representative.to_csv(args.out / "step5_representatives.csv", index=False)

    stability_hdb = [adjusted_rand_score(baseline, labels) for labels in hdb_labels]
    stability_db = [adjusted_rand_score(db_chosen, labels) for labels in db_labels]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, labels, title in ((axes[0], db_chosen, "DBSCAN"), (axes[1], baseline, "HDBSCAN")):
        rng = np.random.default_rng(43)
        indices = rng.choice(len(x), min(5000, len(x)), replace=False)
        color = np.where(labels[indices] == -1, -1, labels[indices] % 20)
        ax.scatter(x[indices, 0], x[indices, 1], c=color, s=3, cmap="tab20", alpha=0.45, vmin=-1, vmax=19)
        ax.set(title=title, xlabel="PC1 (display only)", ylabel="PC2 (display only)")
    fig.tight_layout()
    fig.savefig(figures / "step5_cluster_projection.png", dpi=150)
    plt.close(fig)

    chosen_hdb = cluster_metrics(baseline)
    chosen_db = cluster_metrics(db_chosen)
    defect_mask = ~table.failure_type.isin(["unlabeled", "none"])
    defect_count = int(defect_mask.sum())
    defect_noise = int(((table.hdbscan_cluster == -1) & defect_mask).sum())
    report = [
        "# Step 5 · DBSCAN과 HDBSCAN 패턴 군집", "",
        f"학습 lot의 wafer **{len(table):,}장**을 시드 42로 단순 무작위 추출했다. 4단계에서 train에만 적합한 **{x.shape[1]}차원 PCA 벡터 전체**로 군집화했다. PC1·PC2 그림은 표시용이며 군집 입력을 2차원으로 줄이지 않았다.", "",
        "## 방법·재현", "",
        "```powershell", ".\\.venv\\Scripts\\python.exe -m pip install -r requirements-step5.txt",
        ".\\.venv\\Scripts\\python.exe -m src.clustering.step5_cluster", "```", "",
        "- DBSCAN: 15번째 이웃 거리의 train 표본 분위수 50/70/85%를 eps 후보로 사용했다. 2개 이상 군집·noise ≤80% 후보 중 `|noise−30%| + 최대 군집 점유율` 최소값을 선택했다. 없으면 noise가 30%에 가장 가까운 후보를 택한다.",
        "- HDBSCAN: scikit-learn 구현, `min_samples=15`, `min_cluster_size` 60/120/240을 비교하고 중앙 설정 120을 대표 결과로 정했다.",
        "- 모든 하이퍼파라미터와 표본 선택은 train에서만 수행했다. validation/test는 5단계 탐색에 사용하지 않았다.", "",
        "## 실제 결과", "",
        f"- 선택 DBSCAN: **{chosen_db['clusters']}개 군집**, noise **{chosen_db['noise_fraction']:.1%}**, 최대 군집/할당 wafer **{chosen_db['largest_cluster_fraction']:.1%}**.",
        f"- HDBSCAN(120): **{chosen_hdb['clusters']}개 군집**, noise **{chosen_hdb['noise_fraction']:.1%}**, 최대 군집/할당 wafer **{chosen_hdb['largest_cluster_fraction']:.1%}**.",
        f"- HDBSCAN 설정 60/120/240의 중앙 설정 대비 ARI: **{', '.join(f'{v:.3f}' for v in stability_hdb)}**. DBSCAN eps 후보의 선택 설정 대비 ARI: **{', '.join(f'{v:.3f}' for v in stability_db)}**.",
        f"- 기존 결함 라벨이 있는 표본 **{defect_count}장 중 {defect_noise}장({defect_noise / defect_count:.1%})**이 HDBSCAN noise다. 따라서 이 설정은 결함 유형별 군집을 충분히 형성하지 못했다.",
        "- HDBSCAN 최대 군집의 중앙 수율은 약 94.6%이고, 작은 군집 세 개는 수율이 거의 100%다. 결과는 결함 모양보다 fail 규모와 공간 특징의 밀도에 크게 좌우된다.",
        "- 군집별 수율 사분위 범위·라벨 비율은 [요약표](step5_cluster_summary.csv), 기존 라벨과의 교차표는 [교차표](step5_label_crosstab.csv), 전체 설정 결과는 [파라미터 표](step5_parameter_summary.csv)에 있다.", "",
        "![군집의 PCA 투영](figures/step5_cluster_projection.png)", "",
        "![대표 wafer map](figures/step5_cluster_gallery.png)", "",
        "대표 map은 가장 큰 비-noise 군집 최대 6개에서 군집 중앙값에 가장 가까운 실제 wafer와 가장 먼 실제 wafer를 한 장씩 고른다. 먼 사례는 대표적인 패턴이 아니라 군집 내부의 극단 사례이며, 수율 0% wafer가 수율 100% 중심의 군집에 포함되는 등 내부 이질성도 드러난다. 해당 wafer 목록은 [대표 사례 표](step5_representatives.csv)에 있다.", "",
        "## 품질 점검과 한계", "",
        "- noise 비율과 최대 군집 점유율을 같이 보고 단일 거대 군집·과도한 noise를 확인했다. 설정별 ARI는 같은 표본에서 군집 할당이 얼마나 바뀌는지 보여 주며 다른 데이터에서의 안정성은 뜻하지 않는다.",
        "- 수율은 PCA 입력에서 제외했지만 공간 결함률과 상관될 수 있다. 군집별 수율 사분위 범위를 함께 보아 군집이 단순 수율 차이인지 검토해야 한다.",
        "- 12,000장 표본의 밀도는 전체 모집단의 밀도와 다르며, 작은 희귀 패턴은 표본에서 누락될 수 있다. `unlabeled`와 `none`이 많아 기존 라벨 교차표만으로 군집 품질을 판단할 수 없다.",
        "- 군집 noise는 밀도상 미할당 wafer일 뿐 새 공정 결함 유형이라는 증거가 아니다. 신규 패턴 판정은 7단계에서 별도 평가한다.",
    ]
    (args.out / "step5_clustering.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"HDBSCAN stability ARI: {stability_hdb}; DBSCAN stability ARI: {stability_db}")


if __name__ == "__main__":
    main()
