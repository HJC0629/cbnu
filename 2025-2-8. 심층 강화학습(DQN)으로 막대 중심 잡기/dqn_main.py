import gymnasium as gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import random
from collections import deque
import os


ENV_NAME = "CartPole-v1"
base_dir = os.path.dirname(os.path.abspath(__file__))
model_save_path = os.path.join(base_dir, 'dqn_cartpole.weights.h5')


GAMMA = 0.99  
BATCH_SIZE = 64  
BUFFER_SIZE = 10000  
LEARNING_RATE = 0.001  
EPSILON_START = 1.0  
EPSILON_END = 0.01  
EPSILON_DECAY = 0.995  



class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def put(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):

        batch = random.sample(self.buffer, BATCH_SIZE)

        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer)



def create_q_model(state_shape, action_size):
    model = tf.keras.Sequential([
        layers.Dense(24, activation='relu', input_shape=state_shape),
        layers.Dense(24, activation='relu'),
        layers.Dense(action_size, activation='linear')  
    ])
    return model



class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.epsilon = EPSILON_START


        self.model = create_q_model(state_shape, action_size)  
        self.target_model = create_q_model(state_shape, action_size)  
        self.update_target_model()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        self.buffer = ReplayBuffer(BUFFER_SIZE)

    def update_target_model(self):

        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # 랜덤
        else:
            q_values = self.model.predict(state[np.newaxis], verbose=0)
            return np.argmax(q_values[0])  # Q값이 가장 높은 행동 선택


    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):

        next_q = self.target_model(next_states)
        max_next_q = tf.reduce_max(next_q, axis=1)
        target_q = rewards + (GAMMA * max_next_q * (1 - dones))

        with tf.GradientTape() as tape:

            q_values = self.model(states)


            one_hot_actions = tf.one_hot(actions, self.action_size)
            current_q = tf.reduce_sum(q_values * one_hot_actions, axis=1)


            loss = tf.reduce_mean(tf.square(target_q - current_q))


        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def train(self):
        if self.buffer.size() < BATCH_SIZE:
            return 0  


        states, actions, rewards, next_states, dones = self.buffer.sample()


        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)


        loss = self.train_step(states, actions, rewards, next_states, dones)


        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY

        return loss



if __name__ == "__main__":

    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)

    print(">> 학습 시작 (CartPole)...")
    EPISODES = 150  

    for episode in range(EPISODES):
        state, _ = env.reset()
        score = 0
        done = False

        while not done:

            action = agent.get_action(state)


            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


            if terminated: reward = -10


            agent.buffer.put(state, action, reward, next_state, done)

            state = next_state
            score += 1


            loss = agent.train()


        agent.update_target_model()

        print(f"Episode: {episode + 1}/{EPISODES} | Score: {score} | Epsilon: {agent.epsilon:.2f}")


        if score >= 490:
            print(">> 학습 완료")
            agent.model.save_weights(model_save_path)
            break

    if score < 490:
        agent.model.save_weights(model_save_path)
        print(">> 학습 종료 및 모델 저장 완료")