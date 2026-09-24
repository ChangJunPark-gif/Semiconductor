# 반도체 수율 프로젝트

웨이퍼 수율·공간 패턴 분석과 신규 패턴 탐지를 위한 프로젝트입니다.

구현 범위와 평가 기준은 [프로젝트 명세서](PROJECT_SPEC.md)를 참고하세요.

## 1단계 · 데이터 품질과 수율·bin 분포

원본 [WM-811K](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)를 내려받아 `data/raw/LSWMD.pkl`에 둡니다. 이 컴퓨터의 Codex Python 환경에서 Windows PowerShell로 다음을 실행합니다.

```powershell
$python = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $python -m venv .venv --system-site-packages
.\.venv\Scripts\python.exe -m pip install -r requirements-step1.txt
.\.venv\Scripts\python.exe -c "import kagglehub; kagglehub.dataset_download('qingyi/wm811k-wafer-map', output_dir='data/raw')"
.\.venv\Scripts\python.exe -m src.data.step1_yield
```

분석 결과는 [1단계 보고서](reports/data_report.md)에 정리합니다. wafer별·lot별 CSV 원본은 `data/processed/`에 저장하며 Git에는 올리지 않습니다.

## 2단계 · wafer 공간 시각화

1단계 실행 후 다음 명령으로 패턴별 대표 wafer, 저수율 wafer, lot 내 wafer 순서 그림을 만듭니다.

```powershell
.\.venv\Scripts\python.exe -m src.data.step2_visualize
```

결과와 사례 선정 기준은 [2단계 보고서](reports/step2_visualization.md)에 기록합니다.

## 3단계 · Moran's I와 공간 자기상관

유효 die의 상하좌우 이웃을 기준으로 전체 wafer의 Moran's I를 계산하고, 라벨별로 선정한 사례에 조건부 permutation 검정을 적용합니다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.spatial.step3_moran
```

수치, 검정 범위와 해석상 주의점은 [3단계 보고서](reports/step3_moran.md)에 기록합니다. wafer별 전체 결과 CSV는 `data/processed/`에만 보관합니다.

## 4단계 · 공간 특징과 PCA

전체 wafer의 공간 특징을 추출하고 lot 단위로 train/validation/test를 나눕니다. 결측 대체·표준화·PCA는 train에만 적합합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-step4.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.features.step4_features
```

실제 결과와 한계는 [4단계 보고서](reports/step4_features_pca.md), 각 변수의 정의는 [특징 사전](reports/feature_dictionary.md)에 있습니다. 전체 특징 CSV·PCA 임베딩·변환기는 `data/processed/`에 저장하며 Git에는 올리지 않습니다.

## 5단계 · DBSCAN과 HDBSCAN 군집

학습 lot의 wafer 12,000장을 고정 시드로 추출해 9차원 PCA 공간에서 밀도 기반 군집을 비교합니다. 설정별 군집 수·noise 비율·안정성, 기존 라벨 교차표, 대표 wafer map을 기록합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-step5.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.clustering.step5_cluster
```

[5단계 보고서](reports/step5_clustering.md)에 실제 결과와 한계를 정리했습니다. 웨이퍼별 군집 할당 파일은 `data/processed/`에만 보관합니다.
