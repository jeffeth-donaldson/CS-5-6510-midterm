import cartpole

from time import time
import gym as gym
from gym import wrappers

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

done = False
env = cartpole.CartPoleEnv()
env = wrappers.Monitor(env, "./videos/"+ str(time()) +"cart/")

state = env.reset()
count = 0
while not done:
    action = agent_theta(state)
    next_state, reward, done, _ = env.step(action)
    # x, x_dot, theta, theta_dot
    # print(next_state, count, action)
    state = next_state
    count += 1
    env.render()
   
    

env.close()
