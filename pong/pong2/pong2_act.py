import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import cv2
import matplotlib.pyplot as plt
import imageio
from datetime import datetime


# ----- 하이퍼파라미터 -----
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.02
EPS_DECAY = 200000      #100만->20만, 학습속도 향상을위함
TARGET_UPDATE = 1000
LEARNING_RATE = 1e-4
MEMORY_SIZE = 100000
STACK_SIZE = 4
EPISODES = 1001

# ----- 환경 설정 -----
env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
n_actions = env.action_space.n  # Discrete(6)

# ----- 전처리 -----
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized / 255.0

def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)
    if is_new_episode:
        stacked_frames = deque([frame] * STACK_SIZE, maxlen=STACK_SIZE)
    else:
        stacked_frames.append(frame)
    return np.stack(stacked_frames, axis=0), stacked_frames

# ----- DQN 모델 -----
class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(STACK_SIZE, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# ----- 리플레이 메모리 -----
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# ----- 행동 선택 -----
"""
def select_action(state, policy_net, steps_done, n_actions):
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    if random.random() < eps_threshold:
        return random.randrange(n_actions)
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            return policy_net(state_tensor).argmax().item()
"""
## eval_mode 플래그 추가, gif 로 저장시 앱실론 0 으로 해서 결정적행동만 하게 수정함.

def select_action(state, policy_net, steps_done, n_actions, eval_mode=False):
    if eval_mode:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            return policy_net(state_tensor).argmax().item()
    else:
        eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
        if random.random() < eps_threshold:
            return random.randrange(n_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                return policy_net(state_tensor).argmax().item()

# ----- 시각화: 보상 그래프 저장 -----
def plot_and_save_rewards(rewards, filename="reward_plot.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN Pong Reward over Time")
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.close()
    print(f"✅ Reward plot saved: {filename}")

# ----- 시각화: GIF 저장 -----
def save_gif(env, policy_net, episode, filename="pong_ep{episode}.gif", duration=30):
    frames = []
    state = env.reset()[0]
    stacked_frames = deque([np.zeros((84, 84))] * STACK_SIZE, maxlen=STACK_SIZE)
    state, stacked_frames = stack_frames(stacked_frames, state, True)
    done = False
    while not done:
        frame_rgb = env.render()
        frames.append(frame_rgb)

        #action = select_action(state, policy_net, 0, env.action_space.n)
        # 저장시 실제학습과 괴리가있어서 eval_mode 플래그 추가함.
        action = select_action(state, policy_net, steps_done=0, n_actions=env.action_space.n, eval_mode=True)
        next_state, _, done, _, _ = env.step(action)
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
        state = next_state

    gif_filename = filename.format(episode=episode)
    imageio.mimsave(gif_filename, frames, duration=1 / duration)
    print(f"✅ GIF saved: {gif_filename}")

# ----- 학습 시작 -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(n_actions).to(device)
target_net = DQN(n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(MEMORY_SIZE)

episode_rewards = []
steps_done = 0

for episode in range(EPISODES):
    state = env.reset()[0]
    stacked_frames = deque([np.zeros((84, 84))] * STACK_SIZE, maxlen=STACK_SIZE)
    state, stacked_frames = stack_frames(stacked_frames, state, True)

    total_reward = 0
    done = False
    while not done:
        action = select_action(state, policy_net, steps_done, n_actions)
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward

        next_state_proc, stacked_frames = stack_frames(stacked_frames, next_state, False)
        memory.push((state, action, reward, next_state_proc, done))
        state = next_state_proc
        steps_done += 1

        #if len(memory) >= BATCH_SIZE:
        if len(memory) >= 10000:
            # 배치사이즈 즉 32개서부터하면 너무 적은 샘플로 학습을 시작해서 쓸모없는 학습이 진행됨
            # 만개 샘플이 쌓인뒤에 학습을 시작하도록 변경.
            transitions = memory.sample(BATCH_SIZE)
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

            batch_state = torch.tensor(np.array(batch_state), dtype=torch.float32).to(device)
            batch_next_state = torch.tensor(np.array(batch_next_state), dtype=torch.float32).to(device)
            batch_action = torch.tensor(batch_action, dtype=torch.long).unsqueeze(1).to(device)
            batch_reward = torch.tensor(batch_reward, dtype=torch.float32).to(device)
            batch_done = torch.tensor(batch_done, dtype=torch.float32).to(device)

            q_values = policy_net(batch_state).gather(1, batch_action).squeeze()
            next_q_values = target_net(batch_next_state).max(1)[0]
            expected_q_values = batch_reward + GAMMA * next_q_values * (1 - batch_done)

            loss = nn.MSELoss()(q_values, expected_q_values.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if steps_done % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    episode_rewards.append(total_reward)

    if episode % 10 == 0 and episode > 0:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Episode {episode}, Total Reward: {total_reward}")

    if episode % 50 == 0 and episode > 0:
        plot_and_save_rewards(episode_rewards, filename=f"reward_plot_ep{episode}.png")
        save_gif(env, policy_net, episode)

env.close()
plot_and_save_rewards(episode_rewards, filename="final_reward_plot.png")


# ----- 모델 저장 -----
model_filename = f"policy_net_final.pth"
torch.save(policy_net.state_dict(), model_filename)
print(f"✅ Final model saved: {model_filename}")

target_model_filename = f"target_net_final.pth"
torch.save(target_net.state_dict(), target_model_filename)
print(f"✅ Final target model saved: {target_model_filename}")
