# Autoencoder 기반 제조 설비 이상 탐지 (Anomaly Detection)

## 1. 프로젝트 개요
제조 현장에서는 불량 데이터보다 정상 데이터가 압도적으로 많습니다. 이러한 데이터 불균형 문제를 해결하기 위해 **비지도 학습(Unsupervised Learning)** 기반의 **Deep Autoencoder**를 활용하여 이상 탐지(Anomaly Detection) 모델을 구축했습니다.
본 프로젝트는 심전도(ECG) 데이터를 활용하였으며, 이는 산업 현장의 **진동 센서, 전류 파형, 설비 로그** 등 시계열 센서 데이터의 이상 징후를 포착하는 원리와 동일합니다.

## 2. 개발 환경 및 사용 기술
* **Language:** Python 3.10
* **Framework:** TensorFlow, Keras
* **Algorithm:** Deep Autoencoder (Reconstruction based)
* **Metric:** MAE (Mean Absolute Error)

## 3. 핵심 기술 및 알고리즘

### 3.1 Autoencoder (AE)
입력 데이터를 압축(Encoder)했다가 다시 복원(Decoder)하는 신경망 구조입니다.
* **학습 전략:** 오직 **'정상 데이터'**만 사용하여 모델을 학습시킵니다.
* **이상 탐지 원리:** 모델은 정상 패턴만 학습했기 때문에, 정상 데이터는 잘 복원하지만(Low Error), 학습하지 않은 이상 데이터가 들어오면 복원에 실패하여 오차(High Error)가 커집니다.

### 3.2 Reconstruction Error (복원 오차)
입력값($x$)과 모델이 복원한 값($\hat{x}$) 사이의 차이를 계산하여 이상 점수(Anomaly Score)로 사용합니다.
* **Threshold (임계값) 설정:** 정상 데이터의 복원 오차 분포를 분석하여 `평균 + (표준편차 * n)` 수준으로 설정, 이 값을 초과하면 이상(Anomaly)으로 판정합니다.

## 4. 프로젝트 구조
* **anomaly_main.py**: 데이터 로드, 전처리, Autoencoder 모델 학습 및 이상 탐지 실행
* **anomaly_autoencoder.weights.h5**: 학습된 모델의 가중치 파일
* **anomaly_result_normal.png**: 정상 데이터의 복원 결과 시각화 (성공적인 복원)
* **anomaly_result_abnormal.png**: 이상 데이터의 복원 결과 시각화 (복원 실패로 인한 높은 에러 발생)

## 5. 모델 아키텍처
* **Input:** 140차원 시계열 데이터
* **Encoder:** 140 -> 32 -> 16 -> 8 (차원 축소)
* **Decoder:** 8 -> 16 -> 32 -> 140 (차원 복원)
* **Optimization:** Adam Optimizer, MAE Loss

## 6. 실행 방법

### 6.1 라이브러리 설치
```bash
pip install tensorflow pandas numpy scikit-learn matplotlib