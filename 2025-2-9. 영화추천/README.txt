NCF 기반 영화 추천 시스템 (Neural Collaborative Filtering)
1. 프로젝트 개요
사용자에게 개인화된 콘텐츠를 제공하기 위해 협업 필터링(Collaborative Filtering) 알고리즘을 딥러닝으로 구현한 추천 시스템입니다. 기존의 행렬 분해(Matrix Factorization) 방식을 신경망 구조로 변환하여, 사용자와 아이템(영화) 간의 잠재적 상호작용을 학습하고 사용자별 맞춤 영화를 추천합니다.

2. 개발 환경
Language: Python 3.10

Framework: TensorFlow, Keras

Dataset: MovieLens 100K (스크립트 실행 시 자동 다운로드)

Key Techniques: Embedding Layer, Model Subclassing

3. 핵심 기술 및 구현
3.1 Neural Embeddings
사용자와 영화라는 범주형 데이터를 고정된 크기(50차원)의 연속적인 벡터로 변환하는 임베딩 레이어를 구현했습니다. 이를 통해 희소한(Sparse) 사용자-아이템 행렬을 밀집(Dense)된 벡터 공간으로 매핑하여 학습 효율성을 높였습니다.

3.2 Model Subclassing
Keras의 Sequential 방식이 아닌 tf.keras.Model을 상속받아 RecommenderNet 클래스를 직접 정의했습니다. call 메서드 내에서 임베딩 연산, 편향(Bias) 추가, 내적(Dot Product) 연산을 명시적으로 구현하여 모델의 확장성을 확보했습니다.

3.3 Collaborative Filtering 로직
사용자 벡터와 영화 벡터를 내적(Dot Product)하여 두 벡터 사이의 유사도를 계산하고, 이를 기반으로 평점을 예측합니다. 예측된 평점이 높은 순서대로 영화를 정렬하여 상위 5개를 추천합니다.

4. 프로젝트 구조
recsys_main.py: 데이터 다운로드, 전처리, 모델 클래스 정의, 학습 및 추천 실행 메인 스크립트

recsys_model.weights.h5: 학습이 완료된 모델의 가중치 파일

recsys_loss.png: 학습 및 검증 데이터의 Loss 변화 그래프

ml-latest-small/: MovieLens 데이터셋 폴더

5. 실행 방법
라이브러리 설치
pip install tensorflow pandas numpy matplotlib requests

실행
python recsys_main.py

스크립트를 실행하면 MovieLens 데이터를 자동으로 다운로드하고 전처리를 수행합니다. 이후 10 Epoch 동안 모델 학습을 진행하며, 학습이 완료되면 임의의 사용자에 대한 영화 추천 결과가 터미널에 출력됩니다.

6. 결과 확인
Binary Crossentropy Loss를 사용하여 학습을 진행하였으며, 무작위 추측(Random Guessing) 기준점인 0.69 이하로 Loss가 수렴하는 것을 확인했습니다. 이를 통해 모델이 사용자-영화 간의 패턴을 유의미하게 학습했음을 검증했습니다.