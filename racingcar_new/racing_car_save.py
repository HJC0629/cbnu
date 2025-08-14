import os
import random
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import gym
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import imageio  # GIF 저장을 위해 추가
import datetime  # 시간 출력을 위해 추가

# --- 결과 저장을 위한 디렉토리 생성 ---
if not os.path.exists('car_racing_results'):
    os.makedirs('car_racing_results')


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
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # 96x96 입력 기준, conv3 통과 후 (64, 8, 8) 크기의 특징 맵이 생성됩니다.
        # 따라서 flatten 후의 크기는 64 * 8 * 8 = 4096 입니다.
        self.linear = nn.Linear(64 * 8 * 8, 128)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.linear(x))
        return x


class QNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.feature = CNNFeatureExtractor()
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
    def __init__(self, device='cpu'):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1  # 사용자가 설정한 고정 epsilon 값으로 복원
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 5
        self.device = device

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size).to(self.device)
        self.qnet_target = QNet(self.action_size).to(self.device)
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

    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())


def preprocess(state):
    state = np.transpose(state, (2, 0, 1))  # (H, W, C) → (C, H, W)
    state = state / 255.0
    return state


# --- 추가된 저장 함수들 ---

def save_reward_plot(reward_history, filename):
    """보상 기록을 그래프로 그려 저장합니다."""
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(reward_history)), reward_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward History')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()  # plt 창이 뜨지 않도록 닫아줍니다.


def save_animation_gif(agent, env, filename):
    """에이전트의 플레이를 녹화하여 GIF로 저장합니다."""
    frames = []
    state, info = env.reset()
    done = False

    frames.append(env.render())

    while not done:
        state_p = preprocess(state)
        action = agent.get_action(state_p, use_epsilon=False)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        frames.append(env.render())
        state = next_state

    imageio.mimsave(filename, frames, fps=30)


# --- 메인 학습 로직 ---

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

episodes = 1000
sync_interval = 10
save_interval = 100
save_dir = 'car_racing_results'

env = gym.make('CarRacing-v2', continuous=False, render_mode='rgb_array')
agent = DQNAgent(device=device)

reward_history = []

for episode in range(episodes):
    state, info = env.reset()
    state = preprocess(state)
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = preprocess(next_state)
        done = terminated or truncated

        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if info.get('need_reset', False):
            done = True

    if episode % sync_interval == 0:
        agent.sync_qnet()

    reward_history.append(total_reward)

    # --- 시각을 포함하여 출력하도록 수정 ---
    if episode % 10 == 0:
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{now}] Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}")

    # --- 주기적 저장 및 최종 저장 로직 ---
    is_last_episode = (episode == episodes - 1)
    if (episode + 1) % save_interval == 0 or is_last_episode:
        suffix = 'final' if is_last_episode else f'episode_{episode + 1}'
        print(f"\n--- Saving results for {suffix} ---")

        # 1. 모델 저장
        model_filename = os.path.join(save_dir, f'model_{suffix}.pth')
        torch.save(agent.qnet.state_dict(), model_filename)
        print(f"Model saved to {model_filename}")

        # 2. 보상 그래프 저장
        plot_filename = os.path.join(save_dir, f'reward_history_{suffix}.png')
        save_reward_plot(reward_history, plot_filename)
        print(f"Reward plot saved to {plot_filename}")

        # 3. GIF 애니메이션 저장
        gif_filename = os.path.join(save_dir, f'animation_{suffix}.gif')
        save_animation_gif(agent, env, gif_filename)
        print(f"Animation GIF saved to {gif_filename}\n")

env.close()
print("Training finished.")
