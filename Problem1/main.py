import cartpole

from time import time
import gym as gym
from gym import wrappers

def agent_theta(theta):
    return 1 if theta > 0 else 0

done = False
env = cartpole.CartPoleEnv()
env = wrappers.Monitor(env, "./videos/"+ str(time()) +"cart/")
# env2 = gym.make()
state = env.reset()
count = 0
print(state)

while not done:
    action = agent_theta(state[2])
    next_state, reward, done, _ = env.step(action)
    print(next_state, count, action)
    state = next_state
    count += 1
    env.render()

env.close()
