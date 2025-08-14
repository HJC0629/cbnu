import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import numpy as np
import cv2
from collections import deque
import random

# --- 하이퍼파라미터 ---
STACK_SIZE = 4

# --- 전처리 ---
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

# --- DQN 모델 ---
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

# --- 행동 선택 함수 (결정적) ---
def select_action(state, policy_net, steps_done, n_actions, eval_mode=False):
    if eval_mode:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            return policy_net(state_tensor).argmax().item()
    else:
        eps_threshold = 0.05
        if random.random() < eps_threshold:
            return random.randrange(n_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                return policy_net(state_tensor).argmax().item()

# --- 실행 환경 초기화 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("ALE/Pong-v5", render_mode="human")  # 화면에 출력됨
n_actions = env.action_space.n

# --- 모델 불러오기 ---
policy_net = DQN(n_actions).to(device)
policy_net.load_state_dict(torch.load("target_net_final.pth"))
policy_net.eval()

# --- 게임 플레이 ---
state = env.reset()[0]
stacked_frames = deque([np.zeros((84, 84))] * STACK_SIZE, maxlen=STACK_SIZE)
state, stacked_frames = stack_frames(stacked_frames, state, True)

done = False
total_reward = 0
while not done:
    action = select_action(state, policy_net, steps_done=0, n_actions=n_actions, eval_mode=True)
    next_state, reward, done, _, _ = env.step(action)
    total_reward += reward
    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
    state = next_state

print(f"🎮 게임 종료 - 총 보상: {total_reward}")
env.close()
