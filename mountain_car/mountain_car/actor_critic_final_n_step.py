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

# --- 설정 ---
SEED = 42
MAX_EPISODES = 5000
EARLY_STOP_TARGET = -60.0
EARLY_STOP_WINDOW = 10
ACTOR_LR = 1e-4
CRITIC_LR = 5e-4
GAMMA = 0.98
ENTROPY_COEF = 0.2
SAVE_DIR_MODELS = "models"
SAVE_DIR_OUT = "outputs"
os.makedirs(SAVE_DIR_MODELS, exist_ok=True)
os.makedirs(SAVE_DIR_OUT, exist_ok=True)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# N-step TD를 위한 설정
N_STEPS =10  # N-step TD의 N 값


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


class Actor(nn.Module):
    def __init__(self, state_dim=2, action_dim=3):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return F.softmax(self.fc(x), dim=-1)


class Critic(nn.Module):
    def __init__(self, state_dim=2):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.fc(x)


def improved_shaped_reward(next_state, env_reward, done, env):
    pos, vel = next_state
    shaped = env_reward
    if pos >= env.unwrapped.goal_position:
        shaped += 100.0

    min_pos = env.unwrapped.min_position
    max_pos = env.unwrapped.max_position
    pos_normalized = (pos - min_pos) / (max_pos - min_pos)
    shaped += pos_normalized * 10.0

    max_vel = env.unwrapped.max_speed
    vel_normalized = abs(vel) / max_vel
    shaped += (vel_normalized ** 2) * 5.0
    return shaped


def train(seed=SEED):
    set_seed(seed)
    env = gym.make("MountainCar-v0")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim)
    critic_target = Critic(state_dim)
    critic_target.load_state_dict(critic.state_dict())

    opt_actor = optim.Adam(actor.parameters(), lr=ACTOR_LR)
    opt_critic = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    rewards_history = []

    # N-step TD를 위한 버퍼
    states_buffer = []
    actions_buffer = []
    rewards_buffer = []

    for ep in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            pos, vel = state
            state_normalized = torch.FloatTensor([
                (pos - env.unwrapped.min_position) / (env.unwrapped.max_position - env.unwrapped.min_position),
                (vel - (-env.unwrapped.max_speed)) / (env.unwrapped.max_speed - (-env.unwrapped.max_speed))
            ]).unsqueeze(0)

            probs = actor(state_normalized)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            # 버퍼에 경험 저장
            states_buffer.append(state_normalized.clone().detach())
            actions_buffer.append(action.clone().detach())
            rewards_buffer.append(
                improved_shaped_reward(env.step(int(action.item()))[0], env.step(int(action.item()))[1], done, env))

            # 환경 스텝 진행
            next_state, env_reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated

            # N 스텝만큼 버퍼가 차거나 에피소드가 끝나면 업데이트
            if len(states_buffer) >= N_STEPS or done:
                # TD 타겟 계산
                with torch.no_grad():
                    if done:
                        td_target = torch.zeros(1, 1)
                    else:
                        pos_next, vel_next = next_state
                        next_state_normalized = torch.FloatTensor([
                            (pos_next - env.unwrapped.min_position) / (
                                        env.unwrapped.max_position - env.unwrapped.min_position),
                            (vel_next - (-env.unwrapped.max_speed)) / (
                                        env.unwrapped.max_speed - (-env.unwrapped.max_speed))
                        ]).unsqueeze(0)
                        v_next = critic_target(next_state_normalized)
                        td_target = v_next * (1 - int(done))

                    # N-step 리턴 계산
                    for r in reversed(rewards_buffer):
                        td_target = torch.FloatTensor([[r]]) + GAMMA * td_target

                # Critic 업데이트
                v_curr = critic(states_buffer[0])
                td_error = td_target - v_curr
                critic_loss = td_error.pow(2).mean()
                opt_critic.zero_grad()
                critic_loss.backward()
                opt_critic.step()

                # Actor 업데이트
                # Actor의 log_prob는 그래디언트 계산이 필요하므로 다시 계산해야 합니다.
                # 버퍼에서 꺼낸 상태로 Actor의 forward를 다시 수행합니다.
                log_prob_for_actor = torch.distributions.Categorical(actor(states_buffer[0])).log_prob(
                    actions_buffer[0])
                actor_loss = -(log_prob_for_actor * td_error.detach()).mean()
                opt_actor.zero_grad()
                actor_loss.backward()
                opt_actor.step()

                # 버퍼 초기화
                states_buffer.pop(0)
                actions_buffer.pop(0)
                rewards_buffer.pop(0)

            state = next_state
            ep_reward += env_reward

        rewards_history.append(ep_reward)

        if ep % 100 == 0:
            critic_target.load_state_dict(critic.state_dict())

        if ep % 50 == 0 or ep == 1:
            avg_last = np.mean(rewards_history[-EARLY_STOP_WINDOW:]) if len(
                rewards_history) >= EARLY_STOP_WINDOW else np.mean(rewards_history)
            print(f"Episode {ep}: Avg Reward (last {min(len(rewards_history), EARLY_STOP_WINDOW)}) = {avg_last:.2f}")
            gc.collect()

        if len(rewards_history) >= EARLY_STOP_WINDOW and np.mean(
                rewards_history[-EARLY_STOP_WINDOW:]) >= EARLY_STOP_TARGET:
            print(f"Early stopping at episode {ep} with avg reward {np.mean(rewards_history[-EARLY_STOP_WINDOW:]):.2f}")
            break

    env.close()

    torch.save(actor.state_dict(), os.path.join(SAVE_DIR_MODELS, "actor.pth"))
    torch.save(critic.state_dict(), os.path.join(SAVE_DIR_MODELS, "critic.pth"))
    print(f"Models saved in {SAVE_DIR_MODELS}")

    plt.figure(figsize=(8, 4))
    plt.plot(rewards_history)
    plt.title("Training Rewards (env reward)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR_OUT, "actor_critic_result.png"))
    plt.close()
    print(f"Training reward graph saved to {os.path.join(SAVE_DIR_OUT, 'actor_critic_result.png')}")

    return actor, critic, rewards_history


def evaluate(actor, episodes=10, render_mode=None):
    env = gym.make("MountainCar-v0", render_mode=render_mode)
    total_rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            with torch.no_grad():
                pos, vel = state
                state_normalized = torch.FloatTensor([
                    (pos - env.unwrapped.min_position) / (env.unwrapped.max_position - env.unwrapped.min_position),
                    (vel - (-env.unwrapped.max_speed)) / (env.unwrapped.max_speed - (-env.unwrapped.max_speed))
                ]).unsqueeze(0)

                probs = actor(state_normalized)
                action = torch.argmax(probs, dim=-1).item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            state = next_state
        total_rewards.append(ep_reward)
        print(f"Eval Episode {ep + 1}: Reward = {ep_reward}")
    env.close()

    plt.figure(figsize=(6, 3))
    plt.plot(total_rewards, marker='o')
    plt.title("Evaluation Rewards (deterministic)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR_OUT, "eval_rewards.png"))
    plt.close()
    print(f"Evaluation graph saved to {os.path.join(SAVE_DIR_OUT, 'eval_rewards.png')}")
    return total_rewards


def save_gif(actor, filename=os.path.join(SAVE_DIR_OUT, "actor_behavior.gif"), fps=30):
    try:
        env = gym.make("MountainCar-v0", render_mode="rgb_array")
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
            pos, vel = state
            state_normalized = torch.FloatTensor([
                (pos - env.unwrapped.min_position) / (env.unwrapped.max_position - env.unwrapped.min_position),
                (vel - (-env.unwrapped.max_speed)) / (env.unwrapped.max_speed - (-env.unwrapped.max_speed))
            ]).unsqueeze(0)

            probs = actor(state_normalized)
            action = torch.argmax(probs, dim=-1).item()

        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()
    imageio.mimsave(filename, frames, fps=fps)
    print(f"GIF saved to {filename}")


if __name__ == "__main__":
    actor, critic, history = train()
    evaluate(actor, episodes=10, render_mode=None)
    try:
        save_gif(actor)
    except Exception as e:
        print("GIF 저장 실패: 렌더링 관련 문제. pygame 또는 gymnasium[classic_control] 설치를 권장합니다.")
        print(e)