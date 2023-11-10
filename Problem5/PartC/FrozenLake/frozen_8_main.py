

from time import time
import gym as gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import seaborn as sns
import matplotlib.pyplot as plt
from gym import wrappers
import numpy as np

import random
from agents import SarsaAgent

# env = gym.make('FrozenLake-v1', desc=None, map_name='4x4', is_slippery=True)
# env = gym.make('FrozenLake-v1', desc=None, map_name='8x8', is_slippery=True)
random_map = generate_random_map(size=8)
print(random_map)
env = gym.make('FrozenLake-v1', desc=generate_random_map(size=8))
epochs = 10000
agent = SarsaAgent(env, epochs)

for iteration in range(epochs):
    state = env.reset()

    agent.finalize(iteration)
    done = False

    while not done:

        action = agent.act(state)

        new_state, reward, done, _ = env.step(action)

        agent.update_q(state, new_state, action, reward)
        state = new_state
        # Adding 2% chance of death
        if np.random.rand() < 0.02:
            done = True

        agent.record_data(iteration, action, reward)
        # env.render()

    if done and reward == 1:
        print(iteration, agent.rewards[iteration], agent.paths[iteration])
        # print(agent.paths[iteration])
agent.plot()
env.close()



