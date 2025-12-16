# DQN 기반 CartPole 제어 (Deep Q-Network Control)

## 1. 프로젝트 개요
강화학습(Reinforcement Learning)의 대표적인 알고리즘인 DQN(Deep Q-Network)을 사용하여, 움직이는 카트 위의 막대가 쓰러지지 않도록 중심을 잡는 제어 AI를 구현했습니다.
단순한 라이브러리 호출이 아닌, Replay Buffer, Target Network, Epsilon-Greedy Strategy 등 강화학습의 핵심 로직을 직접 구현하여 엔지니어링 역량을 확보했습니다.

## 2. 개발 환경 및 사용 기술
* Language: Python 3.10
* Framework: TensorFlow, Gymnasium (OpenAI Gym)
* Algorithm: Deep Q-Learning (DQN) with Experience Replay & Fixed Q-Targets
* Visualization: Imageio (GIF Rendering)

## 3. 핵심 기술 및 구현 (Engineering Points)

### 3.1 Custom Replay Buffer (경험 재현 메모리)
* `deque` 자료구조를 활용하여 Agent가 경험한 데이터 `(State, Action, Reward, Next State)`를 저장하고 관리하는 클래스를 직접 구현.
* 데이터 간의 상관관계(Correlation)를 끊고 학습 안정성을 높이기 위해, 저장된 경험 중 무작위로 Mini-batch를 추출하여 학습에 사용.

### 3.2 Target Network (이중 네트워크)
* 학습 대상인 `Main Network`와 정답지 역할을 하는 `Target Network`를 분리.
* 매 스텝마다 업데이트하는 것이 아니라, 일정 주기마다 Main Network의 가중치를 Target Network로 복사(Soft Update)하여 학습의 발산(Divergence)을 방지함.

### 3.3 Bellman Equation Implementation
* `model.fit()`을 사용하지 않고 `tf.GradientTape`를 사용하여 Loss를 직접 계산.
* 수식 구현: $Loss = (R + \gamma \max Q(s', a') - Q(s, a))^2$
* 미래의 예상 보상(Target Q)과 현재의 예측값(Current Q) 사이의 차이를 줄이는 방향으로 역전파(Backpropagation) 수행.

## 4. 프로젝트 구조
* dqn_main.py: DQN Agent 클래스, 학습 루프 구현, 모델 저장
* dqn_visualize.py: 저장된 모델(`.weights.h5`)을 로드하여 시뮬레이션 수행 및 GIF 저장
* cartpole_result.gif: 학습된 AI가 실제로 중심을 잡는 시연 영상

## 5. 실행 방법

### 5.1 라이브러리 설치
```bash
pip install tensorflow gymnasium[classic-control] imageio numpy