# U-Net 기반 표면 결함 분할 (Surface Defect Segmentation)

## 1. 프로젝트 개요
제조 공정 내 표면 스크래치를 검출하기 위한 Semantic Segmentation 모델 구현.
데이터 부족 상황을 가정하여 노이즈 및 선 그리기를 통한 합성 데이터(Synthetic Data) 생성 기술 적용.

## 2. 개발 환경
- Language: Python 3.10
- Framework: TensorFlow, Keras
- Libraries: OpenCV, NumPy, Matplotlib
- Model: U-Net (Custom Implementation)

## 3. 핵심 기술

### 3.1 U-Net Architecture
- Encoder-Decoder 구조를 활용하여 이미지의 특징 추출 및 위치 정보 복원.
- Skip Connection을 구현(Concatenate)하여 Downsampling 과정에서 손실되는 공간 정보를 Upsampling 단계로 전달, 분할 정확도 향상.

### 3.2 Synthetic Data Generation
- OpenCV를 활용하여 임의의 노이즈 배경 및 스크래치 패턴 자동 생성.
- Input Image(Grayscale)와 Mask Image(Binary) 쌍을 실시간으로 생성하여 학습 데이터 확보.

## 4. 프로젝트 구조
- unet_main.py: 데이터 생성, U-Net 모델링, 학습 및 시각화 코드
- unet_model.weights.h5: 학습 완료된 모델 가중치
- segmentation_result.png: 입력 이미지, 정답 마스크, 예측 마스크 비교 결과

## 5. 실행 방법

### 설치
pip install tensorflow opencv-python matplotlib

### 실행
python unet_main.py

스크립트 실행 시 2,000장의 합성 데이터를 생성 후 학습을 진행하며, 결과 이미지는 segmentation_result.png로 저장됨.

## 6. 결과
- 노이즈가 많은 배경에서도 미세한 스크래치 영역을 정확하게 분할(Segmentation)함.
- 픽셀 단위의 결함 검출이 가능함을 확인.