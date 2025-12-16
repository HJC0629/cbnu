# YOLOv8 기반 산업 현장 안전모 착용 감지 시스템 (Safety Helmet Detection)

## 1. 프로젝트 개요
제조 현장의 안전 관리를 자동화하기 위해, 작업자의 **안전모(Hard Hat) 착용 여부를 실시간으로 탐지**하는 객체 탐지(Object Detection) 모델을 구축했습니다.
최신 객체 탐지 알고리즘인 **YOLOv8(You Only Look Once)**을 사용하여 빠른 추론 속도와 높은 정확도를 확보하였으며, 소량의 데이터로도 높은 성능을 내는 **전이 학습(Transfer Learning)** 기법을 적용했습니다.

## 2. 개발 환경 및 사용 기술
* **Language:** Python 3.10
* **Framework:** PyTorch, Ultralytics YOLOv8
* **Data Source:** Roboflow Universe (Hard Hat Workers Dataset)
* **Visualization:** Matplotlib, OpenCV

## 3. 핵심 기술 및 알고리즘

### 3.1 YOLOv8 (You Only Look Once)
* **One-Stage Detector:** 이미지 내의 객체 위치(Localization)와 종류(Classification)를 한 번의 연산으로 동시에 수행하여 실시간 탐지가 가능합니다.
* **Anchor-Free:** 복잡한 앵커 박스 계산을 제거하여 모델 구조를 단순화하고 일반화 성능을 높였습니다.

### 3.2 전이 학습 (Transfer Learning)
* COCO 데이터셋(일반 사물)으로 사전 학습된(Pre-trained) `yolov8n.pt` 모델의 가중치를 가져와, 안전모 데이터셋에 맞게 미세 조정(Fine-tuning)했습니다.
* 이를 통해 학습 시간을 단축시키고(약 20 Epoch), 적은 데이터로도 높은 mAP(mean Average Precision)를 달성했습니다.

## 4. 프로젝트 구조
* **helmet_train.py**: 사전 학습된 YOLOv8 모델 로드 및 사용자 데이터셋 학습 실행
* **helmet_predict.py**: 학습된 가중치(`best.pt`)를 활용한 테스트 이미지 추론 및 시각화
* **data.yaml**: 학습/검증/테스트 데이터셋 경로 및 클래스 정의 파일
* **runs/detect/train/**: 학습 결과 로그, 가중치 파일, 성능 그래프 자동 저장 경로

## 5. 모델 성능 (Evaluation)
* **Confusion Matrix:**
    * **Helmet(안전모):** 93%의 높은 재현율(Recall) 달성.
    * **Head(미착용):** 93%의 정확도로 미착용자를 식별하여 안전 관리 효용성 입증.
    * **Person(사람):** 데이터셋의 라벨 불균형으로 인해 낮은 성능을 보였으나, 핵심 목표인 '안전모 착용 여부' 식별에는 영향 없음.
* **F1-Score:** 0.9 이상 (Confidence 0.4~0.6 구간)

## 6. 실행 방법

### 6.1 라이브러리 설치
```bash
pip install ultralytics opencv-python matplotlib