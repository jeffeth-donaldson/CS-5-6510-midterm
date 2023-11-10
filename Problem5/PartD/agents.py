import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class SarsaAgent(object):

    def __init__(self, environment, iterations = 500, q_table=None):
        self.env = environment

        self.alpha = 0.1            # Learning rate
        self.gamma = 0.9            # Discount factor
        self.epsilon = 0.891        # Exploration rate
        self.min_learning = 0.1
        self.min_explore = 0.1
        self.decay = iterations // 10
        self.iterations = iterations

        self.state_space = self.env.observation_space

        self.upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50) / 1.]
        self.lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50) / 1.]


        self.actions = environment.action_space.n
        
        self.q_table = np.zeros((self.state_space, self.actions))
        self.policy = np.zeros([self.state_space])
        self.steps = np.zeros(self.iterations)
        self.rewards = np.zeros(self.iterations)
        self.paths = dict()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.q_table[state])
        return action

    def update_q(self, state, next_state, action, reward):
        # print(np.max(self.q_table[next_state]))
        self.q_table[state, action] = self.q_table[state, action] + (
            self.alpha * (reward + self.gamma * \
                np.max(self.q_table[next_state]) - \
                self.q_table[state, action]))
        # print(self.q_table[state, action])

    def update_q_8x8(self, state, next_state, action, reward):
        pass

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
        self.epsilon = max(self.min_explore, min(1., 1. - math.log10((iteration + 1) / self.decay)))
        # self.alpha = max(self.min_learning, min(1., 1. - math.log10((iteration + 1) / self.decay)))

    def plot(self):
        sns.lineplot(range(len(self.steps)), self.rewards)
        plt.plot(self.rewards, linewidth=.25)
        plt.xlabel("Iteration")
        plt.ylabel("Solved")
        plt.title("Completed 8x8")
        plt.show()
        print(self.q_table)
        print("Number of reached goal: ", np.count_nonzero(self.rewards >= 1))

    def video_render(self, iteration):
        return self.rewards[iteration] == 1

    def record_data(self, iteration, action, reward):
        self.steps[iteration] += 1
        self.rewards[iteration] += reward
        if iteration not in self.paths.keys():
            self.paths[iteration] = [action]
            return
        self.paths[iteration].append(action)

class DQN(nn.Module):
    def __init__(self, env, observation_space, action_space, device):
        super(DQN, self).__init__()
        self.env = env

        self.fc1 = nn.Linear(observation_space, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear()
        self.fc4 = nn.Linear()

        self.batch_size = 64
        self.gamma = 0.99
        self.eps_start = 1.0
        self.epsilon = self.eps_start
        self.eps_min = 0.05
        self.eps_decay = 0.9997
        self.tau = 0.005
        self.lr = 1e-4

        self.policy_net = DQN(observation_space=observation_space, action_space=action_space).to(device)
        self.target_net = DQN(observation_space=observation_space, action_space=action_space).to(device)

        self.target_net.load_state_dict(self.policy_net.parameters())

    def act(self, state, policy_net):
        if np.random.rand() < self.epsilon:
            return torch.Tensor([[self.env.action_space.sample()]])
        
    def learn(self, memory, batch_size, policy_net, target_net, criterion, optimizer, gamma, device):
        if len(memory) < batch_size:
            return
        
        transitions = memory.sample(batch_size)

        batch = transitions
