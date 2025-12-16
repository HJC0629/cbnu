딥러닝 기반 한국어 스팸 문자 필터링 시스템 (NLP)
1. 프로젝트 개요
TensorFlow와 한국어 형태소 분석기(Kiwi)를 활용하여 SMS 문자 메시지가 스팸인지 정상인지 판별하는 자연어 처리(NLP) 프로젝트입니다. 단순한 키워드 매칭 방식의 한계를 극복하기 위해, 문장의 형태소를 분석하여 실질적인 의미(어근)를 추출한 후 딥러닝 임베딩(Embedding) 레이어를 통해 학습하는 방식을 적용했습니다.

2. 개발 환경 및 사용 기술
Language: Python 3.10

Deep Learning Framework: TensorFlow, Keras

Morphological Analyzer: Kiwipiepy (Kiwi)

Data Processing: Pandas, Numpy, Scikit-learn

Serialization: Pickle

3. 핵심 기술 및 알고리즘
3.1 데이터 전처리 (Morphological Analysis)
한국어의 교착어 특성(어미, 조사의 변화)을 처리하기 위해 Kiwi 형태소 분석기를 도입했습니다.

문장에서 조사와 특수문자를 제거하고 명사(N), 동사/형용사(V), 어근(XR) 등 핵심 의미를 가진 품사만 추출하여 정규화했습니다.

예시: "수익을 보장해드립니다" -> "수익 보장 드리다"

3.2 모델 아키텍처
Input Layer: 텍스트 시퀀스 입력 (Padding 적용)

Embedding Layer: 단어를 고차원 벡터로 변환하여 단어 간의 의미적 관계 학습

GlobalAveragePooling1D: 문장 내 특징 벡터의 평균 추출

Dense Layer: ReLU 활성화 함수를 사용한 은닉층

Output Layer: Sigmoid 활성화 함수를 사용하여 0(정상)과 1(스팸) 사이의 확률 출력

4. 프로젝트 구조
spam_data.csv: 학습 및 검증에 사용되는 데이터셋 (Text, Label 구조)

spam_filter_main.py: 데이터 로드, 전처리(형태소 분석), 모델 학습 및 저장 수행

predict_spam.py: 저장된 모델을 로드하여 사용자 입력 문장에 대한 실시간 스팸 판별 수행

spam_model.h5: 학습 완료된 딥러닝 모델 파일

spam_tokenizer.pkl: 텍스트를 정수 인덱스로 변환하기 위한 토크나이저 객체

5. 데이터셋 구성
Source: 자체 구축 데이터셋 (금융 사기, 도박 광고, 스미싱, 일상 대화 등)

Split: 학습(Train) 80%, 검증(Test) 20%

Stratify: 스팸과 정상 메시지의 비율이 한쪽으로 치우치지 않도록 계층적 분리 적용