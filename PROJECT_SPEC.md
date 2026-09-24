# 웨이퍼 수율·불량 패턴 분석

> **프로젝트 명세서 · v1.0**
>
> Wafer bin map에서 수율과 공간 패턴을 분석하고, 알려진 패턴을 분류하며, 새로운 패턴을 검토 대상으로 선별한다.

| 항목 | 내용 |
| :--- | :--- |
| **목표 직무** | 반도체 전공정 공정·수율 엔지니어 |
| **주요 데이터** | WM-811K wafer bin map |
| **핵심 분석** | 수율 → 공간 시각화 → Moran's I → 특징/PCA → 군집 → 분류 → OOD |
| **최종 결과물** | 재현 가능한 코드, 분석 보고서, 모델 평가, 엔지니어 검토 사례 카드 |
| **예상 기간** | 8주 · CNN 중심 MVP부터 구현 |

### 목차

- [프로젝트 목표](#1-프로젝트-목표)
- [데이터와 해석 범위](#2-데이터와-해석-범위)
- [사용자 시나리오](#3-사용자-시나리오)
- [분석 파이프라인](#4-분석-파이프라인)
- [실험 설계](#5-실험-설계와-누출-방지)
- [기술 구성](#6-기술-구성과-저장소-구조)
- [일정과 우선순위](#7-우선순위와-일정예상-8주)
- [완료 기준](#8-성공-기준과-포트폴리오-표현)

---

## 1. 프로젝트 목표

**만들 것:** wafer bin map의 수율과 공간 패턴을 정량화하고, 알려진 패턴을 분류하며, 학습하지 않은 패턴을 엔지니어 검토 대상으로 올리는 분석 파이프라인.

**답할 질문**

1. 수율이 낮은 웨이퍼의 fail die는 무작위로 흩어졌는가?
2. 같은 형태의 공간 패턴이 여러 wafer 또는 lot에서 반복되는가?
3. 기존 분류기가 설명하지 못하는 새로운 형태인가?

**최종 산출물:** 실행 가능한 코드와 재현 방법, 데이터 품질 보고서, wafer map 시각화, 통계·군집·분류·OOD 평가 보고서, 엔지니어 검토용 사례 카드 5개 이상.

이 프로젝트는 참고 논문의 모델 수치를 재현하는 과제가 아니다. 논문의 분류 모델 비교를 출발점으로 삼되, 통계적 공간 분석과 미지 패턴 탐지까지 연결하는 독립적인 분석 프로젝트다. 참고 논문은 ResNet/CNN 등의 성능과 계산 비용·일반화 한계를 논의한다. 논문에 인용된 최고 정확도를 이 프로젝트의 목표치로 사용하지 않는다. [참고 논문](https://link.springer.com/article/10.1007/s10845-024-02521-0)

## 2. 데이터와 해석 범위

### 2.1 1차 데이터

| 구분 | 명세 |
| :--- | :--- |
| **데이터셋** | 실제 제조 환경에서 수집된 WM-811K: wafer map 811,457개. 원본을 읽어 확인한 고유 lot은 46,293개다. 공개 파일 `LSWMD.pkl` 약 2.1GB. [데이터 카드](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map/metadata) |
| **패턴 라벨** | 일부 샘플에 Center, Donut, Edge-Loc, Edge-Ring, Loc, Random, Scratch, Near-full, none 등이 있다. 실제 라벨 분포는 다운로드 후 검증한다. |
| **원본 필드** | `waferMap`, `lotName`, `waferIndex`, `trianTestLabel`/`failureType` 등 실제 필드명과 값 타입을 먼저 확인한다. |
| **표준 스키마** | `wafer_id`, `lot_id`, `wafer_index`, `map_shape`, `die_state`, `pattern_label`, `split`. 원본→표준 변환 규칙을 기록한다. |
| **die 상태** | 실제 코드를 검사하여 `outside`, `pass`, `fail`로 매핑한다. 바깥/미측정 die는 pass로 세지 않는다. 유효 die가 0개이거나 map이 손상된 샘플은 제외 목록에 남긴다. |
| **저장 정책** | 원본·변환 데이터와 모델 가중치는 Git에서 제외한다. 다운로드·전처리 코드, 체크섬, 사용 조건과 버전만 저장한다. |

검증 결과와 실제 라벨 분포는 `data_report.md`에 남긴다.

### 2.2 해석의 한계

> [!IMPORTANT]
> WM-811K는 최종 전기적 테스트 결과를 공간적으로 나타낸 bin map이다. **이 데이터만으로 전공정의 특정 장비·공정 단계·레시피를 불량 원인으로 단정할 수 없다.**

패턴과 수율은 원인 조사 우선순위를 만드는 신호다. 실제 원인 검증에는 공정 이력, 장비·챔버, lot/시간 순서, 계측·검사 데이터와 전문가 확인이 필요하다. 분류 결과는 “원인” 대신 **“관찰된 패턴”**으로 표현한다.

## 3. 사용자 시나리오

1. 엔지니어가 lot을 선택하면 wafer별 수율과 fail bin 비율, wafer map을 비교한다.
2. 수율이 낮은 wafer에 대해 Moran's I와 공간 특성으로 비무작위 분포 여부를 살핀다.
3. 유사 wafer 군집의 대표 map을 보고, 수동 라벨과 다른 구조나 미분류 map을 확인한다.
4. 분류기는 알려진 패턴의 예측과 확신도를 표시한다. OOD 탐지기가 임계값을 넘긴 경우 분류 결과를 확정하지 않고 `검토 필요`로 표시한다.
5. 사례 카드에는 wafer/lot 식별자, 수율, 공간 통계, 유사 map, 모델 출력, 가능한 조사 방향과 **검증에 필요한 추가 데이터**를 적는다.

## 4. 분석 파이프라인

```mermaid
flowchart LR
    A["① 수율·bin 분포"] --> B["② 공간 시각화"]
    B --> C["③ Moran's I"]
    C --> D["④ 특징 추출·PCA"]
    D --> E["⑤ DBSCAN·HDBSCAN"]
    E --> F["⑥ CNN·ViT 분류"]
    F --> G["⑦ OOD 탐지"]
    G --> H["엔지니어 검토 사례"]
```

| 단계 | 핵심 질문 | 주요 결과 |
| :--- | :--- | :--- |
| 01 · 수율 | 얼마나 많은 유효 die가 통과했는가? | wafer/lot 수율, bin 분포 |
| 02 · 시각화 | fail die가 어디에 있는가? | wafer map 갤러리 |
| 03 · 공간 통계 | 분포가 무작위와 다른가? | Moran's I, permutation 검정 |
| 04 · 특징 | 패턴을 어떻게 수치화할까? | 특징 사전, PCA |
| 05 · 군집 | 비슷한 wafer가 모이는가? | 대표 map, noise·안정성 |
| 06 · 분류 | 알려진 패턴을 구별하는가? | 클래스별 성능·오류 사례 |
| 07 · OOD | 학습하지 않은 패턴을 걸러내는가? | 신규 패턴 후보·오경보율 |

### 01. Yield 및 bin distribution

- **구현:** wafer별 `yield = pass / (pass + fail)`. 결측·무효 die는 분모에서 제외하고 유효 die 수를 함께 표시한다. lot별·패턴별 수율, fail die 비율, 결측 비율을 분석한다. 다중 bin 코드가 있으면 bin별 비율도 계산한다.
- **완료 기준:** `data_report.md`와 wafer/lot 수율 표. 샘플을 수작업으로 계산해 상태 매핑과 분모를 검산한다.

### 02. Wafer spatial visualization

- **구현:** 원래 격자를 유지하고 `outside/pass/fail`을 구분한다. wafer 크기, lot 내 wafer 순서, 패턴별 대표·경계 사례를 비교한다. 모델 입력 리사이즈에는 nearest-neighbor와 유효 영역 mask를 사용한다.
- **완료 기준:** 재현 가능한 시각화 스크립트와 사례 10개 이상. 바깥 영역이 fail로 표시되지 않아야 한다.

### 03. Moran's I: spatial autocorrelation

- **구현:** 유효 die의 4방향 인접 그래프에서 fail 지표의 global Moran's I를 계산한다. 단일 상태 wafer는 `undefined`로 기록한다. 동일 mask와 fail 개수를 고정한 999회 permutation으로 wafer별 귀무분포와 비교한다. 이진 자료용 join-count 통계도 함께 본다.
- **완료 기준:** Moran's I, permutation p-value, 귀무분포와 공간 지표 표. `p<0.05`만으로 원인을 확정하지 않는다. [Moran 가이드](https://pysal.org/esda/stable/user-guide/global_morans_i.html) · [공간 통계 선택 가이드](https://pysal.org/esda/dev/user-guide/global.html)

### 04. PCA / feature engineering

- **구현:** yield, fail 비율, edge/center 및 반경 구간별 fail 비율, 사분면 비대칭, 연결 성분 수·최대 크기, Moran's I와 귀무분포 대비 z-score를 wafer별 특징으로 만든다. 결측 처리·표준화·PCA는 train 데이터에만 적합한다.
- **완료 기준:** 특징 사전, 전처리 파이프라인, PCA 설명 분산·주요 적재량. 군집이 단순 yield 차이만 반영하는지 점검한다.

### 05. DBSCAN / HDBSCAN clustering

- **구현:** wafer별 특징 공간에서 DBSCAN과 HDBSCAN을 비교한다. 군집 대표 map, 내부 다양성, noise 비율, 기존 라벨과의 교차표를 살핀다. 2D 그림용 좌표를 군집 입력으로 재사용하지 않는다.
- **완료 기준:** 군집 요약표와 대표 map 갤러리. 하이퍼파라미터 변화에 대한 안정성, 지나친 noise·단일 군집 여부를 기록한다. noise는 곧바로 새 패턴이 아니다.

### 06. CNN / ViT classification

- **구현:** 라벨이 있는 wafer로 CNN 또는 ResNet을 주 모델로 학습한다. 특징 기반 XGBoost/로지스틱 회귀를 기준선으로 둔다. ViT는 데이터와 연산 자원이 허용될 때 비교한다. 클래스 가중치·증강은 train에만 적용하고 회전·반전의 물리적 타당성을 명시한다.
- **완료 기준:** 테스트 macro-F1, 클래스별 precision/recall, confusion matrix, 추론 시간과 오류 사례. 다수 클래스 정확도만 보고 성공으로 판단하지 않는다.

### 07. 새로운 패턴의 OOD 탐지

- **구현:** 패턴 클래스를 통째로 학습에서 제외하는 `leave-one-class-out` 평가를 반복한다. 임베딩 kNN 거리 또는 Mahalanobis 거리를 novelty score로 사용하고, 최대 softmax 확률을 기준선으로 비교한다. 임계값은 validation에서만 결정한다.
- **완료 기준:** 보류 클래스별 AUROC, AUPR, FPR@95% TPR, 알려진 클래스 오경보율과 사례 그림. 군집 noise와 OOD 점수는 별도로 보고한다. [kNN OOD 원 논문](https://arxiv.org/abs/2204.06507)

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
