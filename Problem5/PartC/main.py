import cartpole
import cartpole_right_weighted as crw

from time import time
import gym as gym
import seaborn as sns
import matplotlib.pyplot as plt
from gym import wrappers
import numpy as np
import math

import random

# Bridget has already taken a reinforcement learning course and
# is using code from that class. The original code was for the
# mountain car gym.
# This tutorial was used to get the q-learning tabular agent working:
# https://medium.com/@flomay/using-q-learning-to-solve-the-cartpole-balancing-problem-c0a7f47d3f9d

def agent_theta(state):
    x = state[0]
    x_dot = state[1]
    theta = state[2]
    theta_dot = state[3]
    if theta_dot > 1:
        return 1
    if theta_dot < -1:
        return 0
    return 1 if theta > 0 else 0

class SarsaAgent(object):
    def __init__(self, environment, iterations = 500):
        self.env = environment

        self.learning_rate = 0.1
        self.discount_rate = 1.0
        self.exploration_rate = 0.2
        self.min_learning = 0.1
        self.min_explore = 0.1
        self.decay = 25
        self.iterations = iterations

        self.state_space = (3, 3, 6, 6)

        self.upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50) / 1.]
        self.lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50) / 1.]

        self.actions = environment.action_space.n
        self.q_table = np.zeros(self.state_space + (self.actions,))
        self.steps = np.zeros(self.iterations)


    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.q_table[state])
        return action

    def update_q(self, state, next_state, action, reward):
        self.q_table[state][action] += (
            self.learning_rate * (reward + self.discount_rate * \
                np.max(self.q_table[next_state]) - \
                self.q_table[state][action]))

    def discrete_values(self, state):
        discretized = list()
        for i in range(len(state)):
            scaling = ((state[i] + abs(self.lower_bounds[i]))
                / (self.upper_bounds[i] - self.lower_bounds[i]))
            new_state = int(round((self.state_space[i] - 1) * scaling))
            new_state = min(self.state_space[i] - 1, max(0, new_state))
            discretized.append(new_state)
        return tuple(discretized)

    def finalize(self, iteration):
        # self.total_reward = 0
        self.exploration_rate = max(self.min_explore, min(1., 1. - math.log10((iteration + 1) / self.decay)))
        self.learning_rate = max(self.min_learning, min(1., 1. - math.log10((iteration + 1) / self.decay)))

    def plot(self):
        # sns.lineplot(range(len(self.steps)), self.steps)
        # plt.xlabel("Iteration")
        # plt.show()
        print("Number of iterations over 200: ", np.count_nonzero(self.steps >= 200))

    def video_render(self, iterations):
        return self.steps[iterations] > 250

# Comment out line 93 and remove comment on line 91 to run regular environment
# env = cartpole.CartPoleEnv()
# Leave line 93 uncommented to run the right bias environment
env = crw.CartPoleEnv()
epochs = 5001
agent = SarsaAgent(env, epochs)
env = wrappers.Monitor(env, "./Problem5/PartC/videos/"+ str(time()) + "cart/", video_callable=lambda x: x % 500 == 0)
# lambda ind: ind % 100 == 0
# agent = NeuralNetworkAgent(env)
for iteration in range(epochs):
    state = agent.discrete_values(env.reset())

    agent.finalize(iteration)
    done = False

    while not done:
        agent.steps[iteration] += 1

        action = agent.act(state)

        next_state, reward, done, _ = env.step(action)
        new_state = agent.discrete_values(next_state)

        agent.update_q(state, new_state, action, reward)
        state = new_state

    if done and agent.steps[iteration] > 200:
        print(iteration, agent.steps[iteration])
    env.render("rgb_array")
agent.plot()
env.close()
