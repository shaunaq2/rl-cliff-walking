import gymnasium as gym
import random
import numpy as np
import pickle
from collections import deque, defaultdict

EPISODES = 30000
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON = 1
EPSILON_DECAY = 0.999

def default_Q_value():
    return 0

if __name__ == "__main__":
    env_name = "CliffWalking-v0"
    env = gym.envs.make(env_name)
    env.reset(seed=1)

    Q_table = defaultdict(default_Q_value)  # starts with a pessimistic estimate of zero reward for each state.
    episode_reward_record = deque(maxlen=100)

    for i in range(EPISODES):
        episode_reward = 0
        done = False
        obs = env.reset()[0]

        if random.uniform(0, 1) < EPSILON:
            action = env.action_space.sample() # Exploration: Random action
        else:
            action = np.argmax([Q_table[(obs, a)] for a in range(env.action_space.n)])

        while not done:
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            if random.uniform(0, 1) < EPSILON:
                next_action = env.action_space.sample()  # Continue exploring randomly
            else:
                next_action = np.argmax([Q_table[(next_obs, a)] for a in range(env.action_space.n)])

            current_q = Q_table[(obs, action)]
            next_q = Q_table[(next_obs, next_action)]
            Q_table[(obs, action)] = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_q - current_q)

            obs = next_obs
            action = next_action

        EPSILON *= EPSILON_DECAY  # Decrease EPSILON as learning progresses
        episode_reward_record.append(episode_reward)

        if i % 100 == 0 and i > 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON))

    model_file = open(f'Q_TABLE_SARSA.pkl', 'wb')
    pickle.dump([Q_table, EPSILON], model_file)
    model_file.close()