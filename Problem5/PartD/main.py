

from time import time
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
from gym import wrappers
import numpy as np
from gymnasium.envs.mujoco import mujoco_env
# from half_cheetah_env import HalfCheetahEnv

from agents import SarsaAgent, DQN
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('HalfCheetah-v4')

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG('MlpPolicy', env, action_noise=action_noise, device=device, verbose=1)
print('Model Created')
model.learn(total_timesteps=15000, log_interval=10)

print("Model Learned")
model.save('saved_model')
print("model Saved")
vec_env = model.get_env()

model = DDPG.load('saved_model')
state = vec_env.reset()
done = False
while not done:
    action, _states = model.predict(state)
    state, rewards, dones, info = vec_env.step(action)
    print(dones[0])
    done = dones[0]
    env.render()

print(rewards, dones)

# if TRAIN:
#     model = DDPG('MlpPolicy', env, action_noise=action_moise, verbose=1)



