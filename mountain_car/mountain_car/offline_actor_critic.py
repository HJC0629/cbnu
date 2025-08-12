import os
import gc
import random
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from collections import deque
import datetime

# 이 라인을 다른 모든 import 문보다 먼저 위치시킵니다.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# --- 설정 ---
SEED = 42
MAX_EPISODES = 10000
EARLY_STOP_TARGET = 90.0
EARLY_STOP_WINDOW = 10
ACTOR_LR = 1e-3
CRITIC_LR = 5e-3
GAMMA = 0.99
TAU = 0.1
BUFFER_SIZE = int(5e4)
BATCH_SIZE = 256
SAVE_DIR_MODELS = "models_ddpg"
SAVE_DIR_OUT = "outputs_ddpg"
os.makedirs(SAVE_DIR_MODELS, exist_ok=True)
os.makedirs(SAVE_DIR_OUT, exist_ok=True)

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용하는 디바이스: {device}")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(np.array(states)).to(self.device),
                torch.FloatTensor(np.array(actions)).to(self.device),
                torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device),
                torch.FloatTensor(np.array(next_states)).to(self.device),
                torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device))

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim=2, action_dim=1):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.fc(x)


class Critic(nn.Module):
    def __init__(self, state_dim=2, action_dim=1):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.fc(torch.cat([state, action], dim=-1))


def train(seed=SEED):
    set_seed(seed)
    env = gym.make("MountainCarContinuous-v0")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = Actor(state_dim, action_dim).to(device)
    critic = Critic(state_dim, action_dim).to(device)
    actor_target = Actor(state_dim, action_dim).to(device)
    critic_target = Critic(state_dim, action_dim).to(device)

    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    opt_actor = optim.Adam(actor.parameters(), lr=ACTOR_LR)
    opt_critic = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    replay_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, device)

    rewards_history = []

    state, _ = env.reset()
    for _ in range(BATCH_SIZE):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        if done:
            state, _ = env.reset()

    for ep in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        ep_reward = 0
        done = False

        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_t = actor(state_t).squeeze(0).cpu().detach().numpy()

            action_t = np.clip(action_t + np.random.normal(0, 0.3), env.action_space.low[0], env.action_space.high[0])

            next_state, reward, terminated, truncated, _ = env.step(action_t)
            done = terminated or truncated

            replay_buffer.add(state, action_t, reward, next_state, done)

            if len(replay_buffer) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = replay_buffer.sample()

                with torch.no_grad():
                    actions_next = actor_target(next_states)
                    q_target_next = critic_target(next_states, actions_next)
                    q_target = rewards + (GAMMA * q_target_next * (1 - dones))

                q_curr = critic(states, actions)
                critic_loss = F.mse_loss(q_curr, q_target)

                opt_critic.zero_grad()
                critic_loss.backward()
                opt_critic.step()

                actions_pred = actor(states)
                actor_loss = -critic(states, actions_pred).mean()

                opt_actor.zero_grad()
                actor_loss.backward()
                opt_actor.step()

                def soft_update(target, source, tau):
                    for target_param, source_param in zip(target.parameters(), source.parameters()):
                        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

                soft_update(critic_target, critic, TAU)
                soft_update(actor_target, actor, TAU)

            state = next_state
            ep_reward += reward

        rewards_history.append(ep_reward)

        if ep % 50 == 0 or ep == 1:
            avg_last = np.mean(rewards_history[-EARLY_STOP_WINDOW:]) if len(
                rewards_history) >= EARLY_STOP_WINDOW else np.mean(rewards_history)

            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[{current_time}] Episode {ep}: Avg Reward (last {min(len(rewards_history), EARLY_STOP_WINDOW)}) = {avg_last:.2f}")

            gc.collect()

        if ep % 500 == 0:
            print(f"Intermediate save at episode {ep}...")
            torch.save(actor.state_dict(), os.path.join(SAVE_DIR_MODELS, f"ddpg_actor_{ep}.pth"))
            torch.save(critic.state_dict(), os.path.join(SAVE_DIR_MODELS, f"ddpg_critic_{ep}.pth"))

            plt.figure(figsize=(8, 4))
            plt.plot(rewards_history)
            plt.title(f"Training Rewards (up to episode {ep})")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR_OUT, f"ddpg_result_{ep}.png"))
            plt.close()

            try:
                eval_rewards = evaluate(actor, episodes=10, render_mode=None)
                plt.figure(figsize=(6, 3))
                plt.plot(eval_rewards, marker='o')
                plt.title(f"Evaluation Rewards (deterministic) at episode {ep}")
                plt.xlabel("Episode")
                plt.ylabel("Reward")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(SAVE_DIR_OUT, f"ddpg_eval_rewards_{ep}.png"))
                plt.close()
                save_gif(actor, filename=os.path.join(SAVE_DIR_OUT, f"ddpg_actor_behavior_{ep}.gif"))
            except Exception as e:
                print(f"Intermediate save failed at episode {ep} due to rendering issues: {e}")

        if len(rewards_history) >= EARLY_STOP_WINDOW and np.mean(
                rewards_history[-EARLY_STOP_WINDOW:]) >= EARLY_STOP_TARGET:
            print(f"Early stopping at episode {ep} with avg reward {np.mean(rewards_history[-EARLY_STOP_WINDOW:]):.2f}")
            break

    env.close()

    torch.save(actor.state_dict(), os.path.join(SAVE_DIR_MODELS, "ddpg_actor_final.pth"))
    torch.save(critic.state_dict(), os.path.join(SAVE_DIR_MODELS, "ddpg_critic_final.pth"))
    print(f"Final models saved in {SAVE_DIR_MODELS}")

    plt.figure(figsize=(8, 4))
    plt.plot(rewards_history)
    plt.title("Training Rewards (env reward)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR_OUT, "ddpg_result_final.png"))
    plt.close()
    print(f"Final training reward graph saved to {os.path.join(SAVE_DIR_OUT, 'ddpg_result_final.png')}")

    return actor, critic, rewards_history


def evaluate(actor, episodes=10, render_mode=None):
    env = gym.make("MountainCarContinuous-v0", render_mode=render_mode)
    total_rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        ep_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                action = actor(torch.FloatTensor(state).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            state = next_state
        total_rewards.append(ep_reward)
        print(f"Eval Episode {ep + 1}: Reward = {ep_reward}")
    env.close()
    return total_rewards


def save_gif(actor, filename=os.path.join(SAVE_DIR_OUT, "ddpg_actor_behavior.gif"), fps=30):
    try:
        env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    except Exception as e:
        print("Render error! Install 'gymnasium[classic_control]' or pygame to enable rendering.")
        raise e

    frames = []
    state, _ = env.reset()
    done = False
    while not done:
        frame = env.render()
        frames.append(frame)
        with torch.no_grad():
            action = actor(torch.FloatTensor(state).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()
    imageio.mimsave(filename, frames, fps=fps)
    print(f"GIF saved to {filename}")


if __name__ == "__main__":
    actor, critic, history = train()
    print("\n--- Final Evaluation ---")
    final_eval_rewards = evaluate(actor, episodes=10, render_mode=None)
    plt.figure(figsize=(6, 3))
    plt.plot(final_eval_rewards, marker='o')
    plt.title("Final Evaluation Rewards (deterministic)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR_OUT, "ddpg_eval_rewards_final.png"))
    plt.close()
    print(f"Final evaluation graph saved to {os.path.join(SAVE_DIR_OUT, 'ddpg_eval_rewards_final.png')}")
    try:
        save_gif(actor, filename=os.path.join(SAVE_DIR_OUT, "ddpg_actor_behavior_final.gif"))
    except Exception as e:
        print("Final GIF save failed: rendering issue.")
        print(e)