import gymnasium as gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import imageio
import os


ENV_NAME = "CartPole-v1"
base_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_dir, 'dqn_cartpole.weights.h5')
save_gif_path = os.path.join(base_dir, 'cartpole_result.gif')


def create_q_model(state_shape, action_size):
    model = tf.keras.Sequential([
        layers.Dense(24, activation='relu', input_shape=state_shape),
        layers.Dense(24, activation='relu'),
        layers.Dense(action_size, activation='linear')
    ])
    return model


env = gym.make(ENV_NAME, render_mode='rgb_array')
state_shape = env.observation_space.shape
action_size = env.action_space.n


model = create_q_model(state_shape, action_size)
model.load_weights(model_path)


frames = []
state, _ = env.reset()
done = False
score = 0

print(">> 시뮬레이션 및 녹화 시작...")

while not done:

    frame = env.render()
    frames.append(frame)


    q_values = model.predict(state[np.newaxis], verbose=0)
    action = np.argmax(q_values[0]) 


    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    state = next_state
    score += 1

    # 너무 길어지면(500프레임) 강제 종료
    if score >= 500:
        break

env.close()
print(f">> 시뮬레이션 종료! 점수: {score}")


imageio.mimsave(save_gif_path, frames, duration=0.02, loop=0)
print(f">> 저장 완료: {save_gif_path}")