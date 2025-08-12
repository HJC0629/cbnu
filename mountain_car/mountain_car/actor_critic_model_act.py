import torch
import gymnasium as gym
import imageio
import os
#from models import Actor  # 학습에 사용한 Actor 클래스 불러오기


import torch.nn as nn
import torch.nn.functional as F
import torch

class Actor(nn.Module):
    def __init__(self, state_dim=2, action_dim=3, hidden_dim=128):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)



# 저장된 모델 경로
actor_model_path = "models/actor.pth"

# 환경 생성
env = gym.make("MountainCar-v0", render_mode="rgb_array")

# 모델 불러오기
actor = Actor()
actor.load_state_dict(torch.load(actor_model_path))
actor.eval()

# 평가 및 애니메이션 저장
def evaluate_and_save_gif(actor, filename="outputs/actor_critic_agent.gif"):
    frames = []
    total_reward = 0
    state, _ = env.reset(seed=42)

    for _ in range(200):  # 최대 타임스텝
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = actor(state_tensor)
        action = torch.argmax(probs, dim=-1).item()

        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        frame = env.render()
        frames.append(frame)

        if terminated or truncated:
            break

    env.close()

    # GIF 저장
    os.makedirs("outputs", exist_ok=True)
    imageio.mimsave(filename, frames, fps=30)
    print(f"🎥 GIF 저장 완료: {filename}")
    print(f"🎯 총 보상: {total_reward}")

evaluate_and_save_gif(actor)
