# 웨이퍼 수율·불량 패턴 분석 및 신규 패턴 탐지 프로젝트 명세서

## 1. 프로젝트 개요

- **목적:** wafer bin map의 수율과 공간 패턴을 정량화하고, 알려진 패턴을 분류하며, 학습하지 않은 패턴을 엔지니어 검토 대상으로 올리는 분석 파이프라인을 만든다.
- **대상 직무:** 반도체 전공정 공정/수율 엔지니어. 결과를 공정 이상 조사에 사용할 수 있도록 수율 변화, 위치, 반복성, 불확실성을 함께 보여준다.
- **핵심 질문:** 수율이 낮은 웨이퍼의 fail die가 무작위로 흩어졌는가? 특정 공간 패턴이 반복되는가? 기존 패턴 분류기가 설명하지 못하는 새 형태인가?
- **최종 산출물:** 재현 가능한 코드와 실행 방법, 데이터 품질 보고서, wafer map 시각화, 통계·군집·분류·OOD 성능 보고서, 엔지니어 검토용 사례 카드 5개 이상.

이 프로젝트는 참고 논문의 모델 수치를 재현하는 과제가 아니다. 논문의 분류 모델 비교를 출발점으로 삼되, 통계적 공간 분석과 미지 패턴 탐지까지 연결하는 독립적인 분석 프로젝트다. 참고 논문은 ResNet/CNN 등의 성능과 계산 비용·일반화 한계를 논의한다. 논문에 인용된 최고 정확도를 이 프로젝트의 목표치로 사용하지 않는다. [참고 논문](https://link.springer.com/article/10.1007/s10845-024-02521-0)

## 2. 데이터와 해석 범위

### 2.1 1차 데이터

- **WM-811K:** 실제 제조 환경에서 수집된 wafer map 811,457개, lot 46,393개를 포함한다. 공개 파일 `LSWMD.pkl`은 약 2.1GB이며, Center, Donut, Edge-Loc, Edge-Ring, Loc, Random, Scratch, Near-full, none 등의 패턴 라벨이 일부 샘플에 있다. 원본 데이터와 라벨 분포는 다운로드 후 검증하여 `data_report.md`에 기록한다. [데이터 카드](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map/metadata)
- **필수 필드:** `waferMap`, `lotName`, `waferIndex`, `trianTestLabel`/`failureType` 등 실제 파일의 필드명을 먼저 검사한다. 분석용 표준 스키마는 `wafer_id`, `lot_id`, `wafer_index`, `map_shape`, `die_state`, `pattern_label`, `split`로 변환한다. 원본 필드명과 변환 규칙을 문서화한다.
- **값 정의:** dataset의 die 상태 코드를 확인하고 `outside`, `pass`, `fail`로 매핑한다. wafer 바깥과 측정되지 않은 die를 pass로 세지 않는다. 유효 die가 0개이거나 map이 손상된 샘플은 제외 목록에 남긴다.
- **저장:** 원본 2.1GB 파일, 변환 데이터, 모델 가중치는 Git에 올리지 않는다. 다운로드·전처리 스크립트와 체크섬, 사용 조건, 버전 정보만 저장한다.

### 2.2 해석의 한계

WM-811K는 최종 전기적 테스트 결과를 공간적으로 나타낸 bin map이다. **전공정의 특정 장비·공정 단계·레시피가 원인이라고 이 데이터만으로 단정할 수 없다.** 패턴과 수율은 원인 조사 우선순위를 만드는 신호다. 실제 원인 검증에는 공정 이력, 장비·챔버, lot/시간 순서, 계측/검사 데이터, 수리 이력과 전문가 확인이 필요하다. 분류 결과는 “원인” 대신 “관찰된 패턴”으로 표현한다.

## 3. 사용자 시나리오

1. 엔지니어가 lot을 선택하면 wafer별 수율과 fail bin 비율, wafer map을 비교한다.
2. 수율이 낮은 wafer에 대해 Moran's I와 공간 특성으로 비무작위 분포 여부를 살핀다.
3. 유사 wafer 군집의 대표 map을 보고, 수동 라벨과 다른 구조나 미분류 map을 확인한다.
4. 분류기는 알려진 패턴의 예측과 확신도를 표시한다. OOD 탐지기가 임계값을 넘긴 경우 분류 결과를 확정하지 않고 `검토 필요`로 표시한다.
5. 사례 카드에는 wafer/lot 식별자, 수율, 공간 통계, 유사 map, 모델 출력, 가능한 조사 방향과 **검증에 필요한 추가 데이터**를 적는다.

## 4. 분석 단계와 완료 기준

| 단계 | 구현 내용 | 산출물·완료 기준 |
| --- | --- | --- |
| **1. Yield 및 bin distribution** | `yield = pass / (pass + fail)`을 wafer 단위로 계산한다. 결측/무효 die는 분모에서 제외하고 유효 die 수를 함께 표시한다. lot별 수율 분포, 클래스별 수율, fail die 비율과 결측 비율을 분석한다. 실제 데이터에 다중 bin 코드가 있으면 bin별 비율을 추가한다. | `data_report.md`, wafer/lot 수율 표. 분모·상태 매핑을 샘플 수작업 검산으로 확인한다. |
| **2. Spatial visualization** | 원래 격자를 유지한 wafer map을 `outside/pass/fail`로 색상 분리한다. wafer 크기별 패싯, lot 내 wafer 순서, 패턴별 대표/경계 사례를 그린다. 분류 입력을 리사이즈할 때는 nearest-neighbor와 유효 영역 mask를 사용한다. | 재현 가능한 시각화 스크립트와 10개 이상 사례. 바깥 영역이 fail처럼 표시되지 않는지 확인한다. |
| **3. Moran's I** | 유효 die만 노드로 사용해 4방향 인접 그래프를 만들고, fail 지표의 global Moran's I를 계산한다. 단일 상태만 있는 wafer는 통계량을 `undefined`로 처리한다. wafer별 fail 비율·격자 크기에 따른 기준 차이를 보정하기 위해 동일 mask와 fail 개수를 고정한 999회 permutation의 귀무분포를 비교한다. 이진 자료에 적합한 join-count 통계도 보조 지표로 비교한다. | Moran's I, permutation p-value, null 분포와 공간 지표 요약표. `p<0.05`만으로 원인을 확정하지 않는다. [PySAL Moran 가이드](https://pysal.org/esda/stable/user-guide/global_morans_i.html), [PySAL 공간 통계 선택 가이드](https://pysal.org/esda/dev/user-guide/global.html) |
| **4. PCA / feature engineering** | wafer별 특징을 만든다: yield, fail 비율, edge/center fail 비율, 반경 구간별 fail 비율, 사분면 비대칭, 연결 성분 수·최대 크기, Moran's I와 귀무분포 대비 z-score. 학습 세트로만 결측 처리·표준화·PCA를 적합한다. PCA는 군집용 차원 축소와 해석용 적재량 확인에 사용한다. | 특징 사전, train에 적합한 전처리 파이프라인, PCA 설명 분산과 주요 적재량. 단순 yield만으로 군집이 갈리는지 점검한다. |
| **5. DBSCAN / HDBSCAN clustering** | wafer 단위 특징 공간에서 군집화한다. DBSCAN을 기준선으로, HDBSCAN을 가변 밀도 대안으로 비교한다. 군집 대표 map과 내부 다양성, noise 비율, 기존 라벨과의 교차표를 검토한다. 임베딩이 아니라 2D 시각화 좌표에서 군집하지 않는다. | 군집 요약표와 대표 map 갤러리. seed/하이퍼파라미터 변화에 대한 군집 안정성, 지나친 noise 또는 단일 군집 여부를 기록한다. noise를 곧바로 새 패턴으로 간주하지 않는다. |
| **6. CNN / ViT classification** | 라벨이 있는 wafer에 대해 간단한 CNN 또는 ResNet 계열을 주 모델로 학습한다. 특성 기반 XGBoost/로지스틱 회귀를 해석 가능한 기준선으로 둔다. ViT는 데이터·연산 자원이 허용되면 비교 실험으로 추가한다. 훈련 데이터에만 클래스 가중치/증강을 적용하고, 회전·반전이 물리적으로 허용되는지 명시한다. | 테스트 macro-F1, 클래스별 precision/recall, confusion matrix, 추론 시간. 다수 클래스 정확도만 보고 성공으로 판단하지 않는다. 라벨 불균형과 모델별 학습 조건을 함께 공개한다. |
| **7. OOD 탐지** | 알려진 클래스 일부를 학습에서 완전히 제외하는 `leave-one-class-out` 실험을 반복한다. 주 모델의 임베딩 kNN 거리 또는 Mahalanobis 거리로 novelty score를 만들고, softmax 최대 확률을 기준선으로 비교한다. 임계값은 validation 세트에서만 결정한다. | 보류 클래스별 AUROC, AUPR, FPR@95% TPR, 알려진 클래스의 오경보율, 사례 시각화. 군집 noise와 OOD 점수를 별도 결과로 보고한다. [kNN OOD 원 논문](https://arxiv.org/abs/2204.06507) |

## 5. 실험 설계와 누출 방지

- `lot_id` 기준으로 train/validation/test를 분리하여 같은 lot의 wafer가 여러 세트에 섞이지 않게 한다. 클래스별 lot 수가 부족하면 이를 명시하고 반복 group split을 사용한다.
- wafer map 해시와 크기·fail mask로 완전 중복/유사 중복을 검사한다. 증강된 샘플은 원본과 같은 split에 둔다.
- 모든 스케일러, PCA, 임계값, 클래스 가중치, 하이퍼파라미터는 train 또는 validation에서만 정한다. 테스트는 최종 평가에만 사용한다.
- 알려진 패턴 분류 평가는 **해당 클래스가 학습에 포함된 closed-set 실험**, 신규 패턴 탐지는 **클래스를 통째로 보류한 open-set 실험**으로 구분한다.
- 데이터가 단일 시점·단일 공개 출처이므로 외부 공정/장비로의 일반화는 검증되지 않았다고 보고한다.

## 6. 기술 구성과 저장소 구조

- **언어:** Python. `pandas`, `numpy`, `matplotlib`/`seaborn`, `scikit-learn`, `PySAL esda`, `PyTorch`를 기본 후보로 사용한다. HDBSCAN은 사용 라이브러리와 버전을 고정한다.
- **재현성:** Python/패키지 버전, 난수 시드, 원본 데이터 해시, split에 배정된 lot 목록, 실험 설정, 모델 체크포인트를 기록한다.
- **권장 구조:** `src/data/`, `src/features/`, `src/models/`, `src/evaluation/`, `configs/`, `reports/`, `notebooks/`, `tests/`.
- **명령형 실행 흐름:** `download → validate → preprocess → split → features → cluster → train → evaluate → report`. 각 단계는 앞 단계 산출물을 읽고 독립적으로 재실행 가능하게 만든다.
- **대시보드(선택):** Streamlit 등으로 lot/wafer 필터, 공간 지도, 군집 대표, 분류/OOD 결과를 탐색한다. 대시보드보다 분석의 재현성과 평가를 먼저 완성한다.

## 7. 우선순위와 일정(예상 8주)

| 기간 | 구현 목표 | 통과 기준 |
| --- | --- | --- |
| 1–2주 | 데이터 입수·검증, yield/bin 분석, wafer map | 상태 매핑·분모 검산, 결측/중복 보고서 |
| 3주 | Moran's I와 permutation, 공간 특징 | 무작위/인위적 edge·center 예제에서 기대 방향 확인 |
| 4주 | PCA와 DBSCAN/HDBSCAN | 대표 map과 군집 안정성 보고서 |
| 5–6주 | 기준선과 CNN 분류 | lot 기준 테스트 성능·오류 사례 확보 |
| 7주 | OOD 실험 | 보류 클래스별 성능·오경보율 확보 |
| 8주 | 사례 카드, README, 발표 자료 | 재현 명령과 한계·후속 조사 제안 정리 |

**MVP:** 1–5단계와 CNN 기준 모델을 완료하고, 7단계는 최소 1개 보류 클래스의 OOD 기준선을 구현한다. ViT와 대시보드는 핵심 평가가 완료된 후 확장한다.

## 8. 성공 기준과 포트폴리오 표현

프로젝트의 성공은 특정 정확도 숫자가 아니라 **재현 가능한 데이터 파이프라인, 누출 없는 평가, 오류 분석, 공정 엔지니어가 검토할 수 있는 설명**으로 판단한다. 최종 발표에는 다음 비교를 반드시 포함한다.

1. 수율만 사용한 기준선과 공간 특성을 추가한 모델의 차이.
2. 특징 기반 모델과 CNN의 macro-F1, 소수 클래스 recall, 추론 비용 차이.
3. known-class 정확도와 unknown-class 탐지 성능의 차이.
4. 잘 분류된 사례, 오류 사례, OOD 사례의 wafer map과 추가 확인이 필요한 공정 데이터.

면접에서는 “전공정 불량 원인을 AI로 찾아냈다”보다 “wafer test bin map으로 비정상 공간 패턴을 정량화하고 신규 패턴을 검토 대상으로 선별했다. 원인 검증에 필요한 장비·공정 이력을 구분해 제시했다”로 설명한다. 실제 공정 인자와 결합하지 않은 단계에서는 수율 **개선 효과**를 주장하지 않는다.

## 9. 참고 자료

- Taha, *Observational and experimental insights into machine learning-based defect classification in wafers*, Journal of Intelligent Manufacturing. https://link.springer.com/article/10.1007/s10845-024-02521-0
- WM-811K dataset card. https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map/metadata
- PySAL, Global Spatial Autocorrelation with Moran's I. https://pysal.org/esda/stable/user-guide/global_morans_i.html
- Sun et al., *Out-of-Distribution Detection with Deep Nearest Neighbors*. https://arxiv.org/abs/2204.06507
