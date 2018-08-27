# Author: Taoz
# Date  : 8/26/2018
# Time  : 3:23 PM
# FileName: tmp_backup.py

# Author: Taoz
# Date  : 8/25/2018
# Time  : 12:13 PM
# FileName: experiment.py


import time
import gym
import numpy as np
from gym.envs.atari import AtariEnv

#
# rng = np.random.RandomState(123456)
# rom = 'Riverraid-v0'
# ale = gym.make(rom).env.ale
# ale.setInt(b'random_seed', rng.randint(1000))
#


# env = gym.make('CartPole-v0')
# env = gym.make('LunarLanderContinuous-v2')
# env = gym.make('Breakout-ram-v0')
# env = gym.make('Alien-ram-v0')

env_name = 'MsPacman-ram-v0'
env_name = 'Centipede-ram-v0'
env_name = 'Assault-ram-v0'
env_name = 'Riverraid-ram-v0'

# ms_pacman
env = AtariEnv(game='riverraid', obs_type='ram', frameskip=4, repeat_action_probability=0.25)

total_reward = 0
for i_episode in range(1):
    observation = env.reset()
    print('++++++++++++ reset ++++++++++++++')
    for t in range(2000):
        env.render()
        # action = env.action_space.sample()
        # if t % 2 == 0:
        #     action = 1
        # else:
        #     action = 11

        action = env.action_space.sample()

        # if action > 4:
        #     action = action % 2
        time.sleep(0.03)

        observation, reward, done, info = env.step(action)
        # reward = env.ale.act(action)
        # done = env.ale.game_over()
        # info = env.ale.lives()
        print("------", action, reward, done, info)
        if reward != 0:
            print('reword [%f] !!!' % reward)
            total_reward += reward
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break


print('total reword: %f' % total_reward)
env.close()


