"""Step 7: leave-one-class-out novelty detection on wafer spatial features."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.step2_visualize import CMAP, NORM
from src.features.step4_features import PCA_FEATURES
from src.models.step6_classify import LABELS, cross_split_keep


INPUT_FEATURES = ["fail_rate", *PCA_FEATURES]


def class_mahalanobis(train: np.ndarray, train_labels: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Minimum class-center Mahalanobis distance with train-only pooled shrinkage covariance."""
    classes = np.unique(train_labels)
    centers = np.vstack([np.mean(train[train_labels == label], axis=0) for label in classes])
    residuals = train - centers[np.searchsorted(classes, train_labels)]
    covariance = LedoitWolf().fit(residuals).covariance_
    precision = np.linalg.inv(covariance + 0.01 * np.eye(covariance.shape[0]))
    distances = []
    for center in centers:
        difference = query - center
        distances.append(np.einsum("ij,jk,ik->i", difference, precision, difference))
    return np.min(np.vstack(distances), axis=0)


def fpr_at_tpr(y_true: np.ndarray, scores: np.ndarray, target: float = 0.95) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    matching = fpr[tpr >= target]
    return float(matching.min()) if len(matching) else 1.0


def score_metrics(known_val: np.ndarray, test_scores: np.ndarray, is_ood: np.ndarray) -> dict[str, float]:
    """Validation-only 5% known false-alarm threshold; final test metrics."""
    threshold = float(np.quantile(known_val, 0.95))
    alarms = test_scores > threshold
    return {
        "auroc": float(roc_auc_score(is_ood, test_scores)),
        "aupr": float(average_precision_score(is_ood, test_scores)),
        "fpr_at_95_tpr": fpr_at_tpr(is_ood, test_scores),
        "validation_threshold": threshold,
        "test_known_false_alarm_rate": float(alarms[~is_ood].mean()),
        "test_heldout_detection_rate": float(alarms[is_ood].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("data/processed/wafer_features.csv"))
    parser.add_argument("--map-cache", type=Path, default=Path("data/processed/step6_labeled_maps.npz"))
    parser.add_argument("--report-output", type=Path, default=Path("reports"))
    args = parser.parse_args()
    args.report_output.mkdir(parents=True, exist_ok=True)
    figures = args.report_output / "figures"
    figures.mkdir(exist_ok=True)

    frame = pd.read_csv(args.features, usecols=["source_row", "lot_id", "failure_type", "split", *INPUT_FEATURES])
    frame = frame[frame.failure_type.isin(LABELS)].copy().reset_index(drop=True)
    stored = np.load(args.map_cache)
    if not np.array_equal(stored["source_rows"], frame.source_row.to_numpy()):
        raise ValueError("Step 6 map cache does not match feature rows")
    keep, removed = cross_split_keep(stored["hashes"], frame.split.to_numpy())
    maps = stored["maps"][keep]
    frame = frame.loc[keep].reset_index(drop=True)
    split = frame.split.to_numpy()
    labels = frame.failure_type.to_numpy()
    train = split == "train"
    validation = split == "validation"
    test = split == "test"
    if any(len(set(frame.lot_id[split == a]) & set(frame.lot_id[split == b])) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise AssertionError("Lot leakage between splits")

    rows = []
    example_scores = {}
    for heldout in LABELS:
        known = labels != heldout
        fit_mask = train & known
        val_mask = validation & known
        test_mask = test  # held-out pattern is tested against every known class
        if not fit_mask.any() or not val_mask.any() or not (test_mask & ~known).any():
            raise ValueError(f"Missing train/validation/test examples for {heldout}")
        pipeline = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                             ("scaler", StandardScaler()),
                             ("classifier", LogisticRegression(class_weight="balanced", max_iter=500, random_state=42))])
        pipeline.fit(frame.loc[fit_mask, INPUT_FEATURES], labels[fit_mask])
        preprocessor = pipeline[:-1]
        train_x = preprocessor.transform(frame.loc[fit_mask, INPUT_FEATURES])
        val_x = preprocessor.transform(frame.loc[val_mask, INPUT_FEATURES])
        test_x = preprocessor.transform(frame.loc[test_mask, INPUT_FEATURES])
        distance_val = class_mahalanobis(train_x, labels[fit_mask], val_x)
        distance_test = class_mahalanobis(train_x, labels[fit_mask], test_x)
        msp_val = 1 - pipeline.predict_proba(frame.loc[val_mask, INPUT_FEATURES]).max(axis=1)
        msp_test = 1 - pipeline.predict_proba(frame.loc[test_mask, INPUT_FEATURES]).max(axis=1)
        truth = labels[test_mask] == heldout
        for method, val_scores, test_scores in (("mahalanobis", distance_val, distance_test),
                                                 ("one_minus_max_softmax", msp_val, msp_test)):
            metrics = score_metrics(val_scores, test_scores, truth)
            rows.append({"heldout_class": heldout, "method": method,
                         "known_train": int(fit_mask.sum()), "known_validation": int(val_mask.sum()),
                         "known_test": int((test_mask & known).sum()), "heldout_test": int(truth.sum()),
                         **metrics})
        example_scores[heldout] = distance_test
        print(f"{heldout}: Mahalanobis AUROC={rows[-2]['auroc']:.3f}, MSP AUROC={rows[-1]['auroc']:.3f}", flush=True)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.report_output / "step7_ood_metrics.csv", index=False)
    comparison = metrics.pivot(index="heldout_class", columns="method", values="auroc").loc[LABELS]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    pos = np.arange(len(LABELS))
    ax.bar(pos - 0.2, comparison["mahalanobis"], width=0.4, label="Mahalanobis")
    ax.bar(pos + 0.2, comparison["one_minus_max_softmax"], width=0.4, label="1 - max softmax")
    ax.set(xticks=pos, xticklabels=LABELS, ylabel="Test AUROC", ylim=(0, 1), title="Leave-one-class-out OOD detection")
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "step7_auroc_comparison.png", dpi=150)
    plt.close(fig)

    # Display one easy and one hard hold-out class, each with low/high novelty examples.
    distance_results = metrics[metrics.method == "mahalanobis"].set_index("heldout_class")
    illustrated = [distance_results.auroc.idxmax(), distance_results.auroc.idxmin()]
    test_indices = np.flatnonzero(test)
    cases = []
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    for row, heldout in enumerate(illustrated):
        scores = example_scores[heldout]
        candidates = np.flatnonzero(labels[test] == heldout)
        for column, (role, choice) in enumerate((("low score", candidates[np.argmin(scores[candidates])]),
                                                 ("high score", candidates[np.argmax(scores[candidates])]))):
            index = test_indices[choice]
            ax = axes[row, column]
            ax.imshow(maps[index], cmap=CMAP, norm=NORM, interpolation="nearest")
            ax.set_title(f"{heldout} · {role}\nscore {scores[choice]:.2f}")
            ax.axis("off")
            cases.append({"heldout_class": heldout, "role": role, "source_row": int(frame.source_row.iloc[index]),
                          "novelty_score": float(scores[choice])})
    fig.suptitle("Held-out pattern examples: Mahalanobis novelty")
    fig.tight_layout()
    fig.savefig(figures / "step7_ood_cases.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(cases).to_csv(args.report_output / "step7_ood_cases.csv", index=False)

    mean = metrics.groupby("method")[["auroc", "aupr", "fpr_at_95_tpr", "test_known_false_alarm_rate", "test_heldout_detection_rate"]].mean()
    heldout_rows = metrics[metrics.method == "mahalanobis"].set_index("heldout_class")
    report = ["# Step 7 · 보류 결함 유형의 OOD 탐지", "",
              "## 실험 설계와 재현", "",
              "```powershell", ".\\.venv\\Scripts\\python.exe -m pip install -r requirements-step7.txt",
              ".\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v",
              ".\\.venv\\Scripts\\python.exe -m src.evaluation.step7_ood", "```", "",
              "결함 라벨 8종을 하나씩 학습에서 완전히 보류했다. 각 실험에서 나머지 7종의 **train**만으로 결측 대체·표준화·가중 로지스틱 회귀를 적합한다. 같은 7종 train의 클래스 중심과 Ledoit–Wolf 수축 공분산(+ 대각선 ridge 0.01)으로 최소 Mahalanobis 제곱거리를 계산한다. 기준선은 같은 분류기의 `1 − 최대 softmax 확률`이다. 수율에 직접 대응하는 `fail_rate`와 4단계 공간 특징 11개가 두 방법의 공통 입력이다.", "",
              "알려진 7종의 validation 점수 **95백분위수**를 각 실험·방법별 경보 임계값으로 정했다. 보류 유형의 validation 사례는 임계값 결정에 사용하지 않았다. Test에서 보류 유형은 양성, 나머지 7종은 음성이다. AUROC·AUPR과 FPR@95% TPR은 test 점수의 곡선 지표이며, 고정된 validation 임계값에서의 실제 경보율과 구분한다.", "",
              f"4단계 lot 분할을 유지하고 6단계의 원본 map 완전 중복 제거 규칙을 다시 적용했다(validation {removed['validation']}장, test {removed['test']}장 제거). `none`·미라벨 wafer는 실험에 포함하지 않았다. 난수 시드는 42다. 원본 해시는 [6단계 보고서](step6_classification.md)에 있다.", "",
              "## 실제 결과", "",
              "| 방법 | 보류 8종 평균 AUROC | 평균 AUPR | 평균 FPR@95% TPR | 알려진 test 오경보율¹ | 보류 test 탐지율¹ |",
              "| :--- | ---: | ---: | ---: | ---: | ---: |",
              *[f"| {method} | {mean.loc[method, 'auroc']:.3f} | {mean.loc[method, 'aupr']:.3f} | {mean.loc[method, 'fpr_at_95_tpr']:.3f} | {mean.loc[method, 'test_known_false_alarm_rate']:.1%} | {mean.loc[method, 'test_heldout_detection_rate']:.1%} |" for method in ("mahalanobis", "one_minus_max_softmax")],
              "", "¹ Validation의 알려진 유형 점수 95백분위수로 고정한 임계값에서 측정했다.", "",
              "Mahalanobis는 평균 AUROC에서 기준선보다 높았지만 고정 임계값에서는 보류 패턴을 평균 36.5%만 검출했다. 특히 `Edge-Ring` 탐지율 0.5%, `Scratch` 0.0%로 실무 경보에 쓰기 어렵다. AUROC만으로 성공이라고 판단할 수 없다.", "",
              "| 보류 유형 | Test 양성 장수 | Mahalanobis AUROC | AUPR | FPR@95% TPR | 알려진 오경보율 | 보류 탐지율 |",
              "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
              *[f"| {label} | {int(heldout_rows.loc[label, 'heldout_test'])} | {heldout_rows.loc[label, 'auroc']:.3f} | {heldout_rows.loc[label, 'aupr']:.3f} | {heldout_rows.loc[label, 'fpr_at_95_tpr']:.3f} | {heldout_rows.loc[label, 'test_known_false_alarm_rate']:.1%} | {heldout_rows.loc[label, 'test_heldout_detection_rate']:.1%} |" for label in LABELS],
              "", "전체 16개 실험의 임계값·표본 수·두 방법 지표는 [상세 CSV](step7_ood_metrics.csv)에 있다.", "",
              "![보류 유형별 AUROC](figures/step7_auroc_comparison.png)", "",
              "![OOD 사례](figures/step7_ood_cases.png)", "",
              "높은·낮은 novelty 점수의 실제 보류 wafer를 가장 쉬운 유형과 어려운 유형에서 각각 골랐다. [사례 인덱스](step7_ood_cases.csv)에 원본 행 번호를 기록했다.", "",
              "## 품질 점검과 한계", "",
              "- AUROC는 임계값 전체의 순위 품질이다. 운영 경보 기준에서는 위의 **보류 탐지율과 알려진 오경보율을 함께** 보아야 한다. `Near-full`처럼 test 표본이 작은 유형의 지표는 불안정하다.",
              "- AUPR은 보류 유형의 test 비율에 영향을 받으므로 보류 유형 간 절댓값을 같은 난이도로 해석하면 안 된다. 표의 평균은 wafer 수 가중이 아닌 유형별 산술평균이다.",
              "- 결함 유형 하나를 전부 제외한 **가상 신규 유형** 실험이다. 실제 미지의 공정 결함이 같은 특성 분포를 따르리라는 보장은 없다. `none`/미라벨 입력에 대한 오경보는 측정하지 않았다.",
              "- 두 점수는 공간 요약 특징에 의존한다. 48×48 CNN 잠재 표현이나 원본 map 기반 거리와 비교하지 않았다. 분할 간 완전 동일 map은 제거했지만 유사 중복은 남을 수 있다.",
              "- 5단계 HDBSCAN의 `noise`는 **train 표본의 밀도 기반 미할당**이고, 여기의 OOD 점수는 **보류 유형 test의 신규성**이다. 대상과 정의가 달라 동일하게 취급하지 않는다."]
    (args.report_output / "step7_ood.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(mean.to_string())


if __name__ == "__main__":
    main()
