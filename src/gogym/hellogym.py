# Author: Taoz
# Date  : 7/21/2018
# Time  : 9:13 AM
# FileName: hellogym.py


import time

import gym
from gym import wrappers

# env = gym.make('CartPole-v0')
# env = gym.make('LunarLanderContinuous-v2')
# env = gym.make('Breakout-ram-v0')
# env = gym.make('Alien-ram-v0')

env_name = 'Centipede-ram-v0'
env_name = 'Assault-ram-v0'
env_name = 'Atlantis-v0'
env_name = 'Riverraid-v0'
env_name = 'SpaceInvaders-v0'
env_name = 'Seaquest-v0'
env_name = 'BeamRider-v0'
env_name = 'Pong-v0'
env_name = 'MsPacman-v0'
env_name = 'Breakout-v0'
env = gym.make(env_name)

e = env.env.reset()
outdir = '/tmp/random-agent-results'
# env = wrappers.Monitor(env, directory=outdir, force=True)

print('action space:', env.action_space)
for i_episode in range(1):
    observation = env.reset()
    for t in range(100):
        env.render()
        # action = env.action_space.sample()
        action = 1

        # print('action:', action)

        time.sleep(0.04)
        print('-- ' * 10)
        print('observation type:', type(observation))
        print('observation:', observation)

        observation, reward, done, info = env.step(action)
        # print("------", action, observation, reward, done)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break

print('DONE.')
print('observation:', str(env.observation_space))
print('action_space:', str(env.action_space))
print('action:', str(env.action_space.contains(17)))
print('action:', str(env.action_space.contains(18)))

env.close()
