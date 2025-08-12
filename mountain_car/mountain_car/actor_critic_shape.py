import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import numpy as np
import os
import imageio


# 🎭 Actor
class Actor(nn.Module):
    def __init__(self, state_dim=2, action_dim=3):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)


# 🎭 Critic
class Critic(nn.Module):
    def __init__(self, state_dim=2):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.fc(x)


# 🎯 Reward Shaping 함수
def shaped_reward(state, reward, done):
    pos, vel = state
    shaped = reward
    shaped += abs(pos - (-0.5)) * 2.0  # 위치 보너스
    shaped += vel * 10.0  # 속도 보너스
    if pos >= 0.5:
        shaped += 100.0  # 목표 도달 보너스
    return shaped


# 🏋️‍♂️ 학습 함수
def train_actor_critic(lr_actor=1e-3, lr_critic=1e-3, gamma=0.99, max_episodes=2000):
    env = gym.make("MountainCar-v0")
    actor = Actor()
    critic = Critic()

    optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)

    all_rewards = []

    for episode in range(max_episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = actor(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            shaped = shaped_reward(next_state, reward, done)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            td_target = shaped + gamma * critic(next_state_tensor) * (1 - int(done))
            td_error = td_target - critic(state_tensor)

            critic_loss = td_error.pow(2).mean()
            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()

            actor_loss = -(dist.log_prob(action) * td_error.detach()).mean()
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            state = next_state
            ep_reward += reward  # 원래 보상 기준

        all_rewards.append(ep_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(all_rewards[-10:])
            print(f"Episode {episode+1}: Avg Reward (last 10) = {avg_reward:.2f}")

    os.makedirs("models", exist_ok=True)
    torch.save(actor.state_dict(), "models/actor.pth")
    torch.save(critic.state_dict(), "models/critic.pth")
    env.close()
    return actor, critic


# 📊 평가 함수
def evaluate_agent(actor, episodes=10):
    env = gym.make("MountainCar-v0", render_mode=None)
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
        print(f"Evaluation Episode {ep+1}: Reward = {ep_reward}")
    avg_reward = np.mean(total_rewards)
    print(f"\n✅ 평균 보상 (over {episodes} episodes): {avg_reward:.2f}")
    env.close()


# 📽 GIF 저장 함수
def save_gif(actor, filename="outputs/actor_behavior.gif"):
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    frames = []
    state, _ = env.reset()
    done = False
    while not done:
        frames.append(env.render())
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = actor(state_tensor)
        action = torch.argmax(action_probs).item()
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    imageio.mimsave(filename, frames, duration=0.03)
    print(f"🎞 GIF 저장 완료: {filename}")
    env.close()


# 🎬 메인 실행
if __name__ == "__main__":
    actor, critic = train_actor_critic()
    evaluate_agent(actor, episodes=10)
    save_gif(actor)
