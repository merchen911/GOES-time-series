# sw-framework-v002

기존 연구 코드의 구성 방식(`configs`/`data`/`model`/`exp`)을 기준으로 재구성한 실험 프레임워크입니다.
데이터 경로를 입력받아 데이터셋을 준비하고, 여러 모델을 학습/검증한 뒤 성능을 비교합니다.

GOES proton/xray 예보를 위해 **parquet 다변량 입력**, **반기(year_half) 단위 누수 없는 교차검증 분할**,
**가변 시계열 길이**, **단·다변량 예보 타깃**을 지원합니다. 기존 CSV 경로는 그대로 유지됩니다.

## 주요 기능
- **두 가지 데이터 경로**
  - **CSV**(레거시): `--data_path *.csv` — 비율 기반 split, 시간 단위 그대로.
  - **parquet**(신규): `--data_path *.parquet` 또는 `--channels` — 5분 공통격자 + 반기 term 분할.
- **단변량 / 다변량 입력**: `--channels PATH:COL [...]`로 N개 채널을 공통 5분 격자에 조인.
- **단변량 / 다변량 예보 타깃**: `--target_cols COL [...]` (입력 채널의 부분집합).
- **시계열 교차검증**: 반기(H1/H2) term 태깅 + 순환 k-fold (`--n_fold`, `--fold_numb`) — 구간이 train/val/test에 걸치지 않아 누수 없음.
- **가변 길이**: `--seq_len`(과거 참조), `--pred_len`(예보 지평) — **5분 스텝 단위**.
- `src/model`의 표준 모델 소스를 직접 사용, 다중 모델 성능 비교 CSV 저장.

## 빠른 시작
환경 준비 후(`pip install -r requirements.txt`, netCDF/parquet 라이브러리는 `python3.12` 환경)
루트의 `main.py`로 실행합니다. 다운로드된 데이터는 프로젝트 `data/` 폴더에 있습니다:
`PROC=<...>/SW_framework/data/goes_data/processed`

- **CSV (레거시)**
  `python main.py --data_path /path/to/data.csv --target_col target --models lstm timesnet`
- **단변량 parquet** (proton p_gt10, 7일 과거 → 1일 예보)
  `python main.py --data_path $PROC/kasi_swpc_particle_5m_v02.parquet \
     --target_col p_gt10 --seq_len 2016 --pred_len 288 --n_fold 5 --fold_numb 0 --models lstm`
- **다변량 입력 → 단일 타깃** (proton+xray 입력, proton 예보)
  `python main.py \
     --channels $PROC/kasi_swpc_particle_5m_v02.parquet:p_gt10 \
                $PROC/kasi_swpc_xray_1m_v02.parquet:xrs_long \
     --target_cols p_gt10 --seq_len 864 --pred_len 288 --models lstm timesnet`
- **다변량 입력 → 다중 타깃** (proton·xray 동시 예보)
  `... --channels $PROC/...:p_gt10 $PROC/...:xrs_long --target_cols p_gt10 xrs_long`

## 데이터 포맷
- **CSV**: 필수 타깃 컬럼(`--target_col`), 선택 시간 컬럼(`--time_col`).
- **parquet**: 시간 컬럼 기본값 `time_utc`, `role` 컬럼(기본 `primary` 필터). 값은 flux 원본이며 `--transform log10`(기본)이 5분 평균 **후** 적용됩니다.

## 주요 옵션
| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--data_path` / `--target_col` | (필수) | 단일 채널 지정. `--channels` 없을 때 이 조합이 단일 채널로 합성됨 |
| `--channels PATH:COL [...]` | `None` | 다중 입력 채널. 1개면 단변량 입력 |
| `--target_cols COL [...]` | `[첫 채널 컬럼]` | 예보 타깃(입력 채널의 부분집합). 1개=단변량 출력, ≥2=다변량 출력 |
| `--split_type` | `year_half` | `year_half` / `year` / `ratio` |
| `--n_fold` / `--fold_numb` | `5` / `0` | 순환 k-fold의 분할 수와 폴드 인덱스(0..n_fold-1) |
| `--seq_len` / `--pred_len` | `24` / `1` | 과거 참조 / 예보 길이. **parquet은 5분 스텝**(예: 1일=288, 7일=2016) |
| `--cadence_min` | `5` | parquet 공통격자 간격(분) |
| `--min_bin_count` | `1` | 5분 bin을 유효로 볼 최소 native 샘플 수 |
| `--transform` | `log10` | `none` / `log10` (5분 평균 후 적용) |
| `--role` | `primary` | parquet `role` 필터 |

> **주의**: `--seq_len`/`--pred_len` 기본값(24/1)은 레거시 **시간 단위**입니다. 5분 parquet에서는 스텝 수를 명시하세요(1일=288, 3일=864, 7일=2016).

## 폴더 구조
- `src/configs/config.py`: CLI/실험 설정 파서
- `src/data/loader.py`: 데이터 로딩/윈도우/분할/로더 생성 (CSV·parquet 두 경로, 공통격자 조인)
- `src/model/`: 표준 모델 소스, `build_model` + `StandardForecastAdapter`(타깃 채널 선택)
- `src/exp/lightning_model.py`: 학습/검증 실행 래퍼
- `src/exp/exp.py`: 멀티 모델 실험 및 성능 비교

## 설계 문서
저장소 내 `docs/`에 위치합니다:
- `dataloader-split-{design,plan}.md` — 반기 분할 + parquet DataModule
- `multivar-join-{design,plan}.md` — 다변량 공통격자 조인, 가변 길이·다중 타깃(§9)
- `model-extensibility-design.md` — 모델 확장성(백본/손실/지표 레지스트리) 설계

## 전처리 스크립트
저장소 내 `preprocessing/`:
- `term_split.py` — 반기 term 태깅 + 순환 k-fold 분할
- `count_fold_samples.py` — 폴드별 학습 가능 윈도우 수 집계
- `make_latex_tables.py` — 논문용 결과 표(LaTeX) 생성

## 결과물
- `runs/<run_name>/ckpt/<model>.pt`
- `runs/<run_name>/score/comparison.csv`
