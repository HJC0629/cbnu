import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 환경 생성
env = gym.make("MountainCar-v0")

# 상태 정규화 함수
def normalize_state(state):
    pos_min, pos_max = -1.2, 0.6
    vel_min, vel_max = -0.07, 0.07
    norm_pos = (state[0] - pos_min) / (pos_max - pos_min)
    norm_vel = (state[1] - vel_min) / (vel_max - vel_min)
    return np.array([norm_pos, norm_vel], dtype=np.float32)

# Actor
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

# Critic
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

# 행동 선택
def choose_action(actor, state):
    state = torch.FloatTensor(np.array(state))
    probs = actor(state)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action), probs

# 학습 함수
def train_actor_critic(lr_actor, lr_critic, gamma, max_episodes=2000):
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

            # ✅ Reward shaping
            reward += 10 * abs(next_state[1])  # 속도 기반 보상 추가

            next_state = normalize_state(next_state)

            state_tensor = torch.FloatTensor(state)
            next_state_tensor = torch.FloatTensor(next_state)
            reward_tensor = torch.FloatTensor([reward])

            # Critic 업데이트
            td_target = reward_tensor + gamma * critic(next_state_tensor) * (1 - int(done))
            td_error = td_target - critic(state_tensor)

            critic_loss = td_error.pow(2)
            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()

            # Actor 업데이트 + entropy 보너스
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            actor_loss = -log_prob * td_error.detach() - 0.01 * entropy

            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            state = next_state
            total_reward += reward

            if done:
                episode_rewards.append(total_reward)
                break

        # 학습 중간 출력
        if (episode + 1) % 100 == 0:
            avg_last = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}: Avg Reward (last 10) = {avg_last:.2f}")

    return episode_rewards

# 결과 시각화
def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Improved Actor-Critic on MountainCar-v0")
    plt.grid()
    plt.show()

# 메인 실행
if __name__ == "__main__":
    rewards = train_actor_critic(lr_actor=1e-3, lr_critic=1e-3, gamma=0.99, max_episodes=2000)
    plot_rewards(rewards)
