import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import random


"""학습이 너무안되서 0.01 에서 0.05 로 조정함"""
"""학습 및 모델저장과 그래프 저장"""
"""더이상의 나아짐이 없음 시작위치를 변경하면 나아지겠지만, 다른 모델을 사용해서 테스트 진행"""
# 폴더 생성
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ✅ 시드 고정 함수
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ✅ 상태 정규화
def normalize_state(state):
    pos_min, pos_max = -1.2, 0.6
    vel_min, vel_max = -0.07, 0.07
    norm_pos = (state[0] - pos_min) / (pos_max - pos_min)
    norm_vel = (state[1] - vel_min) / (vel_max - vel_min)
    return np.array([norm_pos, norm_vel], dtype=np.float32)

# ✅ Actor
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, state):
        return self.fc(state)

# ✅ Critic
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, state):
        return self.fc(state)

# ✅ 행동 선택
def choose_action(actor, state):
    state = torch.FloatTensor(np.array(state))
    probs = actor(state)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action), probs

# ✅ 학습
def train_actor_critic(lr_actor, lr_critic, gamma, max_episodes=2000, seed=42):
    set_seed(seed)
    env = gym.make("MountainCar-v0", render_mode=None)
    env.reset(seed=seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim)

    optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)

    episode_rewards = []

    for episode in range(max_episodes):
        state, _ = env.reset()
        state = normalize_state(state)
        total_reward = 0

        while True:
            action, log_prob, probs = choose_action(actor, state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            reward += 10 * abs(next_state[1])  # reward shaping
            next_state = normalize_state(next_state)

            state_tensor = torch.FloatTensor(state)
            next_state_tensor = torch.FloatTensor(next_state)
            reward_tensor = torch.FloatTensor([reward])

            # Critic
            td_target = reward_tensor + gamma * critic(next_state_tensor) * (1 - int(done))
            td_error = td_target - critic(state_tensor)
            critic_loss = td_error.pow(2)

            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()

            # Actor
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            actor_loss = -log_prob * td_error.detach() - 0.05 * entropy

            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            state = next_state
            total_reward += reward

            if done:
                episode_rewards.append(total_reward)
                break

        if (episode + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}: Avg Reward (last 10) = {avg:.2f}")

    env.close()

    # ✅ 모델 저장
    torch.save(actor.state_dict(), "models/actor.pth")
    torch.save(critic.state_dict(), "models/critic.pth")
    print("✅ 모델 저장 완료: models/actor.pth, models/critic.pth")

    return actor, episode_rewards

# ✅ 그래프 저장
def save_plot(rewards, filename="outputs/actor_critic_result.png"):
    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Actor-Critic on MountainCar-v0")
    plt.grid()
    plt.savefig(filename)
    plt.close()
    print(f"📊 그래프 저장 완료: {filename}")

# ✅ 에이전트 애니메이션 저장
def save_agent_animation(actor, filename="outputs/actor_critic_agent.gif"):
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    state, _ = env.reset()
    state = normalize_state(state)
    frames = []

    while True:
        frame = env.render()
        frames.append(frame)
        action, _, _ = choose_action(actor, state)
        next_state, _, terminated, truncated, _ = env.step(action)
        state = normalize_state(next_state)
        if terminated or truncated:
            break

    env.close()
    imageio.mimsave(filename, frames, fps=30)
    print(f"🎥 애니메이션 저장 완료: {filename}")

# ✅ 실행
if __name__ == "__main__":
    actor, rewards = train_actor_critic(lr_actor=1e-3, lr_critic=1e-3, gamma=0.99, max_episodes=2000, seed=42)
    save_plot(rewards)
    save_agent_animation(actor)
