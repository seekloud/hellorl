# Author: Taoz
# Date  : 8/25/2018
# Time  : 12:23 PM
# FileName: environment.py

import gym
from gym.envs.atari import AtariEnv


class GameEnv(object):
    def __init__(self, game, obs_type, frame_skip):
        self.gym_env = AtariEnv(game=game,
                                obs_type=obs_type,
                                frameskip=frame_skip,
                                repeat_action_probability=0.05)

    def step(self, action):
        observation, reward, done, lives = self.gym_env.step(action)
        return observation, reward, done, lives

    def render(self):
        return self.gym_env.render()

    def random_action(self):
        return self.gym_env.action_space.sample()

    def action_num(self):
        return self.gym_env.action_space.n

    def reset(self, skip_begin_frame=5):
        assert skip_begin_frame > 0
        self.gym_env.reset()
        obs = None
        for _ in range(skip_begin_frame):
            obs, _, _, _ = self.gym_env.step(self.gym_env.action_space.sample())
        return obs

    def close(self):
        self.gym_env.close()
