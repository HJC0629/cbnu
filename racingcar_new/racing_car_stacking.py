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
import cv2

# --- 결과 저장을 위한 디렉토리 생성 ---
if not os.path.exists('car_racing_results'):
    os.makedirs('car_racing_results')


# --- 행동 공간을 단순화하고 항상 가속을 추가하는 Wrapper ---
class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)
        # 기존의 복잡한 행동 공간 대신, 3가지 간단한 행동으로 정의
        # 0: 직진, 1: 왼쪽, 2: 오른쪽
        self.action_space = gym.spaces.Discrete(3)

    def action(self, action):
        # 에이전트가 선택한 간단한 행동(0, 1, 2)을
        # 실제 환경의 [조향, 가속, 브레이크] 값으로 변환
        if action == 0:  # 직진
            return [0.0, 0.2, 0.0]  # 조향 0, 약한 가속, 브레이크 0
        elif action == 1:  # 왼쪽
            return [-0.5, 0.2, 0.0]  # 왼쪽으로 조향, 약한 가속
        elif action == 2:  # 오른쪽
            return [0.5, 0.2, 0.0]  # 오른쪽으로 조향, 약한 가속


# --- 최종 전처리 함수 (리사이즈 + 그레이스케일) ---
def preprocess(state):
    state = cv2.resize(state, (84, 84))
    state = np.dot(state[..., :3], [0.2989, 0.5870, 0.1140])
    state = state / 255.0
    return state[np.newaxis, :, :].astype(np.float32)


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
    def __init__(self, input_channels=4):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.linear = nn.Linear(3136, 512)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.linear(x))
        return x


class QNet(nn.Module):
    def __init__(self, action_size, input_channels=4):
        super().__init__()
        self.feature = CNNFeatureExtractor(input_channels)
        self.l1 = nn.Linear(512, 256)
        self.l2 = nn.Linear(256, action_size)

    def forward(self, x):
        x = self.feature(x)
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class DQNAgent:
    def __init__(self, action_size, device='cpu', input_channels=4):
        self.gamma = 0.99
        self.lr = 1e-4
        self.buffer_size = 10000
        self.batch_size = 64  # 배치 사이즈 증가
        self.action_size = action_size  # Wrapper에 맞춰 수정
        self.device = device

        self.epsilon = 1.0
        self.epsilon_decay = 0.999  # Epsilon 감소 속도 약간 증가
        self.epsilon_min = 0.02

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


def save_animation_gif(agent, env, filename):
    frames = []
    state, info = env.reset()

    state_p = preprocess(state)
    state_stack = deque([state_p] * 4, maxlen=4)
    current_stacked_state = np.concatenate(state_stack, axis=0)

    frames.append(env.render())
    done = False
    while not done:
        action = agent.get_action(current_stacked_state, use_epsilon=False)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        frames.append(env.render())

        next_state_p = preprocess(next_state)
        state_stack.append(next_state_p)
        current_stacked_state = np.concatenate(state_stack, axis=0)

        state = next_state

    imageio.mimsave(filename, frames, fps=30)


# --- 메인 학습 로직 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

episodes = 5000
save_interval = 100
save_dir = 'car_racing_results'
input_channels = 4

# --- 환경 생성 및 Wrapper 적용 ---
env_raw = gym.make('CarRacing-v2', continuous=True, render_mode='rgb_array', disable_env_checker=True)
env = ActionWrapper(env_raw)

agent = DQNAgent(action_size=env.action_space.n, device=device, input_channels=input_channels)

reward_history = []
total_steps = 0
sync_interval = 1000  # 타겟 네트워크 동기화 주기 (스텝 기준)

for episode in range(episodes):
    state, info = env.reset()

    state = preprocess(state)
    state_stack = deque([state] * 4, maxlen=4)
    current_stacked_state = np.concatenate(state_stack, axis=0)

    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(current_stacked_state)
        next_state, reward, terminated, truncated, info = env.step(action)

        next_state_p = preprocess(next_state)
        state_stack.append(next_state_p)
        next_stacked_state = np.concatenate(state_stack, axis=0)

        done = terminated or truncated

        agent.update(current_stacked_state, action, reward, next_stacked_state, done)
        current_stacked_state = next_stacked_state
        total_reward += reward
        total_steps += 1

        # 스텝 기준으로 타겟 네트워크 동기화
        if total_steps % sync_interval == 0:
            agent.sync_qnet()

        if info.get('need_reset', False):
            done = True

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

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
        save_animation_gif(agent, env, gif_filename)
        print(f"Animation GIF saved to {gif_filename}\n")

env.close()
print("Training finished.")
