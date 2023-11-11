from time import time
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecVideoRecorder
import gymnasium as gym
import numpy as np
# from half_cheetah_env import HalfCheetahEnv
import half_cheetah_v4 as hc4

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# env = gym.make('HalfCheetah-v4', render_mode="human")
env = hc4.HalfCheetahEnv()

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG('MlpPolicy', env, action_noise=action_noise, device=device, verbose=1)

model.learn(total_timesteps=10000, log_interval=10)

vec_env = model.get_env()

done = False
state = vec_env.reset()
while not done:
    action, _states = model.predict(state)
    state, reward, done, info = vec_env.step(action)
    done = done[0]
vec_env.render('human')
vec_env.close()



