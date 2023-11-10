

from time import time
import gym as gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import seaborn as sns
import matplotlib.pyplot as plt
from gym import wrappers
import numpy as np

import random
from agents import SarsaAgent

env = gym.make('FrozenLake-v1', desc=None, map_name='4x4', is_slippery=True)
# env = gym.make('FrozenLake-v1', desc=None,map_name='8x8', is_slippery=True)
# env = gym.make('FrozenLake-v1', desc=generate_random_map(size=8), is_slippery=True)
epochs = 5000
agent = SarsaAgent(env, epochs)

# env = wrappers.Monitor(env, "./Problem5/PartC/FrozenLake/videos" + str(time()) + "cart/", video_callable= lambda x: x % 100 == 0)

for iteration in range(epochs):
    state = env.reset()

    agent.finalize(iteration)
    done = False

    while not done:

        action = agent.act(state)

        new_state, reward, done, _ = env.step(action)

        agent.update_q(state, new_state, action, reward)
        state = new_state

        agent.record_data(iteration, action, reward)

    if done and reward == 1:
        # print(agent.paths[iteration])
        print(iteration, agent.rewards[iteration], agent.steps[iteration])
        env.render()
agent.plot()
env.close()



