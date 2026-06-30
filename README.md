# sw-framework-v001

기존 연구 코드의 구성 방식(`configs`/`data`/`model`/`exp`)을 기준으로 재구성한 실험 프레임워크입니다.
데이터 경로를 입력받아 데이터셋을 준비하고, 여러 모델을 학습/검증한 뒤 성능을 비교합니다.

## 주요 기능
- CSV 데이터 로드 및 전처리
- 시계열 윈도우 기반 `DataModule`/`DataLoader` 구성
- `src/model`의 표준 모델 소스를 직접 사용
- 학습/검증 루프 및 best model 저장
- 다중 모델 성능 비교 결과 CSV 저장

## 빠른 시작
1. 환경 준비
   - `pip install -r requirements.txt`
2. 실행 예시
   - `python main.py --data_path /path/to/data.csv --target_col target --models lstm timesnet`

## 데이터 포맷
- CSV 파일
- 필수: 타깃 컬럼 (`--target_col`)
- 선택: 시간 컬럼 (`--time_col`, 기본값 없음)
  - 주어지면 시간순 정렬 후 split

## 폴더 구조
- `src/configs/config.py`: CLI/실험 설정 파서
- `src/data/loader.py`: 데이터 로딩/윈도우/분할/로더 생성
- `src/model/`: 표준 모델 소스 및 모델 로더
- `src/exp/lightning_model.py`: 학습/검증 실행 래퍼
- `src/exp/exp.py`: 멀티 모델 실험 및 성능 비교

## 결과물
- `runs/<run_name>/ckpt/<model>.pt`
- `runs/<run_name>/score/comparison.csv`
