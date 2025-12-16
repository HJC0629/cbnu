# LSTM 기반 삼성전자 주가 예측 (Time Series Forecasting)

## 1. 프로젝트 개요
순환 신경망(RNN)의 일종인 LSTM(Long Short-Term Memory) 모델을 활용하여 시계열 데이터(Time Series Data)의 미래 패턴을 예측하는 프로젝트입니다.
본 프로젝트는 주식 시장의 과거 데이터를 분석하여 미래 주가를 예측하는 것을 목표로 하며, 이는 산업 현장의 센서 데이터(온도, 진동, 압력 등)를 기반으로 설비의 이상 징후나 고장 시점을 예측하는 예지보전(Predictive Maintenance) 기술과 동일한 논리 구조를 가집니다.

## 2. 개발 환경 및 사용 기술
* Language: Python 3.10
* Deep Learning Framework: TensorFlow, Keras
* Data Source: yfinance (Yahoo Finance API)
* Data Processing: Pandas, Numpy, Scikit-learn (MinMaxScaler)
* Visualization: Matplotlib

## 3. 핵심 기술 및 알고리즘

### 3.1 LSTM (Long Short-Term Memory)
기존의 DNN(Deep Neural Network)이 현재 시점의 데이터만 독립적으로 처리하는 한계를 극복하기 위해, 과거의 정보(Time Step)를 기억하고 현재의 판단에 반영하는 LSTM 네트워크를 사용했습니다.
* Sequence Learning: 연속된 데이터의 흐름과 패턴을 학습합니다.
* Forget Gate: 불필요한 과거 정보는 잊고, 중요한 정보만 장기 기억 셀(Cell State)에 남겨 예측 정확도를 높입니다.

### 3.2 데이터 전처리
* Normalization: 데이터의 스케일 차이에 따른 학습 저하를 방지하기 위해 `MinMaxScaler`를 사용하여 모든 데이터를 0과 1 사이로 정규화했습니다.
* Windowing: 시계열 데이터를 지도 학습(Supervised Learning) 문제로 변환하기 위해, 과거 50일치 데이터(t-50 ~ t-1)를 입력(X)으로, 다음 날(t)의 데이터를 정답(Y)으로 하는 슬라이딩 윈도우 방식을 적용했습니다.

## 4. 프로젝트 구조
* stock_main.py: 데이터 수집, 전처리, 모델 학습, 예측 및 시각화까지 전 과정을 수행하는 메인 스크립트
* stock_lstm_model.h5: 학습이 완료된 LSTM 모델 파일
* stock_result.png: 실제 주가와 AI 모델의 예측값을 비교한 결과 그래프

## 5. 모델 아키텍처
* Input Layer: (Window Size, 1) 형태의 3차원 텐서 입력
* LSTM Layer 1: 50 유닛, Sequence 반환 (Deep LSTM 구조)
* Dropout Layer: 0.2 (과적합 방지)
* LSTM Layer 2: 50 유닛
* Dense Layer: 최종 1개의 예측값(주가) 출력
