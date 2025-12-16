# DCGAN 기반 합성 데이터 생성 및 증강 (Synthetic Data Generation using DCGAN)

## 1. 프로젝트 개요
제조 현장의 고질적인 문제인 '불량 데이터 부족(Data Scarcity)' 문제를 해결하기 위해, 생성형 AI 기술인 DCGAN(Deep Convolutional Generative Adversarial Network)을 구축하였습니다.
본 프로젝트는 Generator(생성자)와 Discriminator(판별자) 두 신경망이 경쟁하며 학습하는 과정을 통해, 무작위 노이즈로부터 유의미한 패턴(의류 이미지)을 생성하는 것을 목표로 합니다. 이는 향후 결함 데이터를 인위적으로 생성하여 학습 데이터를 증강(Data Augmentation)하는 기술의 기초가 됩니다.

## 2. 개발 환경 및 사용 기술
* Language: Python 3.10
* Framework: TensorFlow, Keras
* Algorithm: DCGAN (Deep Convolutional GAN)
* Core Logic: Custom Training Loop (tf.GradientTape) - 100% Manual Implementation
* Visualization: Matplotlib, Imageio (GIF Creation)

## 3. 핵심 기술 및 알고리즘

### 3.1 Custom Training Loop (사용자 정의 학습 루프)
기존의 `model.fit()` 함수를 사용하지 않고, TensorFlow의 `tf.GradientTape`를 활용하여 학습의 전 과정을 직접 제어했습니다.
* 미분(Gradient) 계산: 손실 함수에 대한 가중치의 기울기를 직접 계산하여 역전파(Backpropagation)를 수행.
* 최적화(Optimizer) 적용: 생성자와 판별자에 각각 별도의 Adam Optimizer를 적용하여 정밀한 학습 제어 구현.

### 3.2 Adversarial Training (적대적 학습)
* Generator: 판별자가 '진짜'라고 착각할 만큼 정교한 가짜 이미지를 생성하도록 학습.
* Discriminator: 생성자가 만든 가짜와 실제 데이터를 정확히 구별하도록 학습.
* Nash Equilibrium: 두 모델이 팽팽하게 경쟁하며 손실(Loss)이 특정 값에 수렴하지 않고 진동하는 균형 상태 도달.

## 4. 프로젝트 구조
* gan_main.py: DCGAN 모델 설계, 사용자 정의 학습 루프 실행, 이미지 생성 및 Loss 그래프 저장
* make_gif.py: 학습 과정에서 생성된 이미지들을 시계열 순서로 묶어 타임랩스(GIF)로 변환
* generated_images/: Epoch 별로 생성된 결과물 이미지 저장 폴더
* gan_loss_graph.png: 생성자와 판별자의 손실 변화 그래프 (학습 안정성 확인용)
* gan_training_process.gif: 노이즈가 의미 있는 이미지로 변화하는 과정을 시각화한 결과물

## 5. 모델 아키텍처
* Generator: Dense -> Reshape -> Conv2DTranspose (Upsampling) -> Tanh Output
* Discriminator: Conv2D (Downsampling + Dropout) -> Flatten -> Dense (Scalar Output)
* Input: 100-dim Random Noise Vector

## 6. 실행 방법

### 6.1 라이브러리 설치
```bash
pip install tensorflow matplotlib numpy imageio