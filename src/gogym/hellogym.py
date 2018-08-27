# Author: Taoz
# Date  : 7/21/2018
# Time  : 9:13 AM
# FileName: hellogym.py


import time

import gym
from gym import wrappers


# env = gym.make('CartPole-v0')
#env = gym.make('LunarLanderContinuous-v2')
#env = gym.make('Breakout-ram-v0')
#env = gym.make('Alien-ram-v0')

env_name = 'MsPacman-ram-v0'
env_name = 'Centipede-ram-v0'
env_name = 'Assault-ram-v0'
env_name = 'Riverraid-ram-v0'
env_name = 'Riverraid-v0'
env = gym.make(env_name)


e = env.env.reset()
outdir = '/tmp/random-agent-results'
# env = wrappers.Monitor(env, directory=outdir, force=True)

for i_episode in range(2):
    observation = env.reset()
    for t in range(50):
        env.render()
        env.action_space.sample()
        # action = env.action_space.sample()
        if t % 2 == 0:
            action = 1
        else:
            action = 11

        # if action > 4:
        #     action = action % 2
        time.sleep(0.04)

        observation, reward, done, info = env.step(action)
        print("------", action, observation, reward, done)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break


print('DONE.')
print('observation:', str(env.observation_space))
print('action:', str(env.action_space))
print('action:', str(env.action_space.n))
print('action:', str(env.action_space.contains(17)))
print('action:', str(env.action_space.contains(18)))

env.close()







