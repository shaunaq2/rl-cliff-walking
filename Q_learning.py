import gymnasium as gym
import random
import numpy as np
import time
from collections import deque
import pickle


from collections import defaultdict


EPISODES =  30000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999


def default_Q_value():
    return 0
def get_action(s, eps, q_tbl, action_space):
    if random.uniform(0, 1) < eps:
        return action_space.sample()
    else:
        return max(range(action_space.n), key = lambda a: q_tbl[(s, a)])

if __name__ == "__main__":
    env_name = "CliffWalking-v0"
    env = gym.envs.make(env_name)
    env.reset(seed=1)

    Q_table = defaultdict(default_Q_value) 
    episode_reward_record = deque(maxlen=100)
    for i in range(EPISODES):
        episode_reward = 0
        done = False
        obs = env.reset()[0]

        action = get_action(obs, EPSILON, Q_table, env.action_space)
    

        while (not done):
            next_state,reward,terminated,truncated,info = env.step(action)
            episode_reward += reward
            
            max_q = max(Q_table[(next_state, a)] for a in range(env.action_space.n))
            Q_table[(obs, action)] = (1-LEARNING_RATE) * Q_table[((obs, action))] + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_q)
            action = get_action(next_state, EPSILON, Q_table, env.action_space)
            obs = next_state


            done = terminated or truncated

        EPSILON *= EPSILON_DECAY
            

        episode_reward_record.append(episode_reward) 
     
        if i % 100 == 0 and i > 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    
    model_file = open(f'Q_TABLE_QLearning.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    model_file.close()
    