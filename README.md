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
