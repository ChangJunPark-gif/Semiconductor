# 프로젝트 진행 현황

기준 문서: [프로젝트 명세서](PROJECT_SPEC.md)

| 단계 | 상태 | 검증 가능한 결과 |
| :--- | :--- | :--- |
| 1. Yield 및 bin distribution | **완료** | [데이터 품질·수율 보고서](reports/data_report.md), [분석 코드](src/data/step1_yield.py) |
| 2. Wafer spatial visualization | 대기 | — |
| 3. Moran's I | 대기 | — |
| 4. PCA / feature engineering | 대기 | — |
| 5. DBSCAN / HDBSCAN | 대기 | — |
| 6. CNN / ViT classification | 대기 | — |
| 7. OOD 탐지 | 대기 | — |

매 실행에서는 가장 앞의 미완료 단계 **하나**를 구현·검증하고, 결과 링크와 실제 한계를 이 표에 기록한다. 원본·변환 데이터는 `data/`에만 보관한다.
