import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import random
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import gym
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import imageio
import datetime

# --- 결과 저장을 위한 디렉토리 생성 ---
if not os.path.exists('car_racing_results'):
    os.makedirs('car_racing_results')


# --- 3단계(프레임 스태킹) 적용 시 주석 해제 ---
# def preprocess(state):
#     """상태를 RGB에서 그레이스케일로 변환하고 정규화합니다."""
#     state = np.dot(state[...,:3], [0.2989, 0.5870, 0.1140])
#     state = state / 255.0
#     # (H, W) -> (1, H, W) 형태로 변환하여 스태킹 준비
#     return state[np.newaxis, :, :].astype(np.float32)

# --- 기존 코드의 전처리 함수 ---
def preprocess(state):
    state = np.transpose(state, (2, 0, 1))  # (H, W, C) → (C, H, W)
    state = state / 255.0
    return state.astype(np.float32)


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)
        state = np.stack([x[0] for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data]).astype(np.int32)
        return state, action, reward, next_state, done


class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels=3):  # 3단계 적용 시 input_channels=4 로 변경
        super().__init__()
        # 입력 채널 수를 인자로 받아 유연하게 대처
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.linear = nn.Linear(64 * 8 * 8, 128)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.linear(x))
        return x


class QNet(nn.Module):
    def __init__(self, action_size, input_channels=3):  # 3단계 적용 시 input_channels=4 로 변경
        super().__init__()
        self.feature = CNNFeatureExtractor(input_channels)
        self.l1 = nn.Linear(128, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = self.feature(x)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DQNAgent:
    def __init__(self, device='cpu', input_channels=3):
        # --- 1단계: 하이퍼파라미터 조정 ---
        self.gamma = 0.99  # GitHub 저장소 값 (기존 0.98)
        self.lr = 1e-4  # GitHub 저장소 값 (기존 0.0005)
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 5
        self.device = device

        # --- 2단계: Epsilon Decay 도입 ---
        self.epsilon = 1.0  # 탐험 확률 초기값
        self.epsilon_decay = 0.999  # 점진적으로 감소시킬 비율
        self.epsilon_min = 0.02  # 탐험 확률 최솟값

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size, input_channels).to(self.device)
        self.qnet_target = QNet(self.action_size, input_channels).to(self.device)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

    def get_action(self, state, use_epsilon=True):
        if use_epsilon and np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = state[np.newaxis, :]
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                qs = self.qnet(state)
            return qs.argmax(dim=1).item()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)

        qs = self.qnet(state)
        q = qs[torch.arange(self.batch_size), action]

        with torch.no_grad():
            next_qs = self.qnet_target(next_state)
            next_q = next_qs.max(dim=1)[0]
            target = reward + (1 - done) * self.gamma * next_q

        loss = F.mse_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon 값을 점진적으로 감소시킴
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())


def save_reward_plot(reward_history, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(reward_history)), reward_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward History')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def save_animation_gif(agent, env, filename, use_frame_stacking=False):
    frames = []
    state, info = env.reset()

    if use_frame_stacking:
        # 프레임 스태킹을 사용할 경우, 초기 상태 스택 구성
        state_p = preprocess(state)
        state_stack = deque([state_p] * 4, maxlen=4)
        current_stacked_state = np.concatenate(state_stack, axis=0)

    frames.append(env.render())
    done = False
    while not done:
        if use_frame_stacking:
            action = agent.get_action(current_stacked_state, use_epsilon=False)
        else:
            state_p = preprocess(state)
            action = agent.get_action(state_p, use_epsilon=False)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        frames.append(env.render())

        if use_frame_stacking:
            next_state_p = preprocess(next_state)
            state_stack.append(next_state_p)
            current_stacked_state = np.concatenate(state_stack, axis=0)

        state = next_state

    imageio.mimsave(filename, frames, fps=30)


# --- 메인 학습 로직 ---
# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

episodes = 1000
sync_interval = 20  # 타겟 네트워크 동기화 주기 조정
save_interval = 100
save_dir = 'car_racing_results'

# --- 3단계 적용 여부 ---
USE_FRAME_STACKING = False  # True로 바꾸면 프레임 스태킹 적용
input_channels = 4 if USE_FRAME_STACKING else 3

env = gym.make('CarRacing-v2', continuous=False, render_mode='rgb_array')
agent = DQNAgent(device=device, input_channels=input_channels)

reward_history = []

for episode in range(episodes):
    state, info = env.reset()

    if USE_FRAME_STACKING:
        # 프레임 스태킹을 위한 초기 상태 구성
        state = preprocess(state)
        state_stack = deque([state] * 4, maxlen=4)
        current_stacked_state = np.concatenate(state_stack, axis=0)
    else:
        current_stacked_state = preprocess(state)

    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(current_stacked_state)
        next_state, reward, terminated, truncated, info = env.step(action)

        if USE_FRAME_STACKING:
            next_state_p = preprocess(next_state)
            state_stack.append(next_state_p)
            next_stacked_state = np.concatenate(state_stack, axis=0)
        else:
            next_stacked_state = preprocess(next_state)

        done = terminated or truncated

        agent.update(current_stacked_state, action, reward, next_stacked_state, done)
        current_stacked_state = next_stacked_state
        total_reward += reward

        if info.get('need_reset', False):
            done = True

    if episode % sync_interval == 0:
        agent.sync_qnet()

    reward_history.append(total_reward)

    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{now}] Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

    is_last_episode = (episode == episodes - 1)
    if (episode + 1) % save_interval == 0 or is_last_episode:
        suffix = 'final' if is_last_episode else f'episode_{episode + 1}'
        print(f"\n--- Saving results for {suffix} ---")
        model_filename = os.path.join(save_dir, f'model_{suffix}.pth')
        torch.save(agent.qnet.state_dict(), model_filename)
        print(f"Model saved to {model_filename}")
        plot_filename = os.path.join(save_dir, f'reward_history_{suffix}.png')
        save_reward_plot(reward_history, plot_filename)
        print(f"Reward plot saved to {plot_filename}")
        gif_filename = os.path.join(save_dir, f'animation_{suffix}.gif')
        save_animation_gif(agent, env, gif_filename, use_frame_stacking=USE_FRAME_STACKING)
        print(f"Animation GIF saved to {gif_filename}\n")

env.close()
print("Training finished.")
