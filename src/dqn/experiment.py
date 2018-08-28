# Author: Taoz
# Date  : 8/25/2018
# Time  : 12:13 PM
# FileName: experiment.py


import time
import gym
import numpy as np
from gym.envs.atari import AtariEnv

from src.dqn.player import Player

from src.dqn.game_env import GameEnv
from src.dqn.replay_buffer import ReplayBuffer
from src.dqn.q_learning import QLearning
from src import utils
import mxnet as mx
import mxnet as nd


class Experiment(object):
    ctx = utils.try_gpu(0)

    GAME_NAME = 'riverraid'
    OBSERVATION_TYPE = 'image'  # image or ram
    FRAME_SKIP = 4
    EPOCH_NUM = 5
    EPOCH_LENGTH = 3000

    PHI_LENGTH = 4
    CHANNEL = 3
    WIDTH = 160
    HEIGHT = 210
    INPUT_SAMPLE = nd.random.uniform(0, 255, (1, PHI_LENGTH * CHANNEL, HEIGHT, WIDTH), ctx=ctx) / 255.0

    # print('input:', type(INPUT_SAMPLE))
    # print('input:', INPUT_SAMPLE)

    BUFFER_MAX = 100000
    DISCOUNT = 0.99
    RANDOM_SEED = int(time.time() * 1000) % 100000000

    mx.random.seed(RANDOM_SEED)
    rng = np.random.RandomState(RANDOM_SEED)

    def __init__(self):
        self.step_count = 0
        self.episode_count = 0
        self.q_learning = QLearning(Experiment.ctx,
                                    Experiment.INPUT_SAMPLE,
                                    Experiment.DISCOUNT)
        self.game = GameEnv(game=self.GAME_NAME,
                            obs_type=self.OBSERVATION_TYPE,
                            frame_skip=self.FRAME_SKIP)
        self.player = Player(self.game, self.q_learning, Experiment.rng)
        self.replay_buffer = ReplayBuffer(Experiment.HEIGHT,
                                          Experiment.WIDTH,
                                          Experiment.CHANNEL,
                                          Experiment.rng,
                                          Experiment.DISCOUNT,
                                          Experiment.BUFFER_MAX)

    def start(self):
        for i in range(1, self.EPOCH_NUM + 1):
            self._run_epoch(i)
        print('experiment done.')
        self.game.close()

    def _run_epoch(self, epoch):
        steps_left = self.EPOCH_LENGTH
        random_action = False
        while steps_left > 0:
            if self.step_count < 1000:
                random_action = True
            ep_steps, ep_reward = self.player.run_episode(epoch, steps_left, self.replay_buffer, render=True,
                                                          random_action=random_action)
            self.step_count += ep_steps
            self.episode_count += 1
            steps_left -= ep_steps
            print('++++++ episode [%d] finish, episode step=%d, total_step=%d'
                  % (self.episode_count, ep_steps, self.step_count))

        print('#####  epoch [%d] finish, episode=%d, step=%d' % (epoch, self.episode_count, self.step_count))


if __name__ == '__main__':
    exper = Experiment()
    exper.start()

    pass
