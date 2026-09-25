# 프로젝트 진행 현황

기준 문서: [프로젝트 명세서](PROJECT_SPEC.md)

| 단계 | 상태 | 검증 가능한 결과 |
| :--- | :--- | :--- |
| 1. Yield 및 bin distribution | **완료** | [데이터 품질·수율 보고서](reports/data_report.md), [분석 코드](src/data/step1_yield.py) |
| 2. Wafer spatial visualization | **완료** | [공간 시각화 보고서](reports/step2_visualization.md), [그림 생성 코드](src/data/step2_visualize.py) |
| 3. Moran's I | **완료** | [공간 자기상관 보고서](reports/step3_moran.md), [전체 계산 코드](src/spatial/step3_moran.py), [검증](tests/test_step3_spatial.py) |
| 4. PCA / feature engineering | **완료** | [특징·PCA 보고서](reports/step4_features_pca.md), [특징 사전](reports/feature_dictionary.md), [분석 코드](src/features/step4_features.py), [검증](tests/test_step4_features.py) |
| 5. DBSCAN / HDBSCAN | **완료** | [군집·안정성 보고서](reports/step5_clustering.md), [대표 map](reports/figures/step5_cluster_gallery.png), [분석 코드](src/clustering/step5_cluster.py), [검증](tests/test_step5_clustering.py) |
| 6. CNN / ViT classification | **완료** | [CNN·기준선 평가 보고서](reports/step6_classification.md), [분류 코드](src/models/step6_classify.py), [지표](reports/step6_metrics.json), [검증](tests/test_step6_classify.py) |
| 7. OOD 탐지 | **완료** | [보류 유형별 OOD 보고서](reports/step7_ood.md), [분석 코드](src/evaluation/step7_ood.py), [전체 지표](reports/step7_ood_metrics.csv), [검증](tests/test_step7_ood.py) |

매 실행에서는 가장 앞의 미완료 단계 **하나**를 구현·검증하고, 결과 링크와 실제 한계를 이 표에 기록한다. 원본·변환 데이터는 `data/`에만 보관한다.

7단계까지 실행·검증을 마쳤다. 추가 자동 실행은 종료한다.
