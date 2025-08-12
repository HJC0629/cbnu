import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import imageio
import os
"""모델 평가 및 애니메이션 재생"""
# 🎭 Actor 정의 (저장할 때와 같은 구조)
class Actor(nn.Module):
    def __init__(self, state_dim=2, action_dim=3):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),  # 입력층 → 첫 번째 hidden layer
            nn.ReLU(),
            nn.Linear(256, 128),        # 첫 번째 hidden layer → 두 번째 hidden layer
            nn.ReLU(),
            nn.Linear(128, action_dim)  # 두 번째 hidden → 출력층
        )

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)


# 🔧 환경 및 모델 로딩
env = gym.make("MountainCar-v0", render_mode="rgb_array")  # 애니메이션을 위한 렌더링 모드

# 모델 인스턴스
actor = Actor()


# 모델 경로
actor_path = "models/actor.pth"
critic_path = "models/critic.pth"

# state_dict 불러오기
actor.load_state_dict(torch.load(actor_path))

actor.eval()


# 📊 평가 함수
def evaluate_agent(actor, episodes=10):
    total_rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = actor(state_tensor)
            action = torch.argmax(action_probs).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            state = next_state

        total_rewards.append(ep_reward)
        print(f"Episode {ep+1}: Reward = {ep_reward}")

    avg_reward = np.mean(total_rewards)
    print(f"\n✅ 평균 보상 (over {episodes} episodes): {avg_reward:.2f}")

# 📽 애니메이션 저장 함수
def save_gif(actor, filename="outputs/actor_behavior_eval.gif"):
    frames = []
    state, _ = env.reset()
    done = False

    while not done:
        frame = env.render()
        frames.append(frame)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = actor(state_tensor)
        action = torch.argmax(action_probs).item()

        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    # 저장 디렉토리 생성
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    imageio.mimsave(filename, frames, duration=0.03)
    print(f"🎞 GIF 저장 완료: {filename}")

# 🎬 실행
if __name__ == "__main__":
    evaluate_agent(actor, episodes=10)
    save_gif(actor)
    env.close()
