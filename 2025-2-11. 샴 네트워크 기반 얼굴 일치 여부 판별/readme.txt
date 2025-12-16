샴 네트워크 기반 얼굴 검증 시스템 (Siamese Face Verification)
1. 프로젝트 개요
두 장의 얼굴 이미지를 비교하여 동일인 여부를 판별하는 One-Shot Learning 기반의 검증(Verification) 모델. CNN을 통해 특징 벡터(Feature Vector)를 추출하고, 유클리드 거리(Euclidean Distance)와 Contrastive Loss를 사용하여 유사도를 학습함.

2. 개발 환경
Language: Python 3.10

Framework: TensorFlow, Keras

Dataset: Olivetti Faces (Sklearn 내장 데이터셋)

Technique: Weight Sharing, Siamese Architecture, Contrastive Loss

3. 핵심 기술
3.1 Siamese Architecture (Weight Sharing)
두 개의 입력(Image A, Image B)이 가중치를 공유하는 동일한 CNN(Base Network)을 통과하도록 설계. 각 이미지에서 추출된 128차원 특징 벡터 간의 거리를 계산하여 유사도 판단.

3.2 Contrastive Loss Function
일반적인 Cross Entropy가 아닌 거리 기반의 손실 함수 직접 구현. 동일인(Label 1)일 경우 거리를 0에 가깝게, 타인(Label 0)일 경우 거리가 Margin 이상 벌어지도록 학습.

4. 프로젝트 구조
siamese_main.py: 데이터 쌍 생성, 모델 구현, 학습 및 검증 시각화

siamese_model.weights.h5: 학습 가중치

siamese_result.png: 테스트 쌍에 대한 유사도 거리(Distance) 예측 시각화

5. 실행 방법
설치
pip install tensorflow scikit-learn matplotlib numpy

실행
python siamese_main.py

스크립트 실행 시 Olivetti Faces 데이터를 로드하고 긍정/부정 쌍을 생성하여 학습을 수행. 결과 이미지는 Distance 점수와 함께 저장됨.

6. 결과
학습된 모델은 동일인의 사진 쌍에 대해 낮은 거리 값(0.5 미만)을, 다른 사람의 쌍에 대해 높은 거리 값을 출력함. 데이터에 없는 새로운 인물이라도 특징 벡터 비교를 통해 재학습 없이 검증 가능.