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
    EPOCH_NUM = 2
    EPOCH_LENGTH = 2000

    PHI_LENGTH = 4
    CHANNEL = 3
    WIDTH = 160
    HEIGHT = 210
    INPUT_SAMPLE = nd.random.uniform(0, 255, (PHI_LENGTH, CHANNEL, HEIGHT, WIDTH), ctx=ctx).astype(np.uint8)

    BUFFER_MAX = 100000
    DISCOUNT = 0.99
    RANDOM_SEED = int(time.time() * 1000)

    mx.random.seed(RANDOM_SEED)
    rng = np.random.RandomState(RANDOM_SEED)


    def __init__(self):
        self.game = GameEnv(game=self.GAME_NAME,
                            obs_type=self.OBSERVATION_TYPE,
                            frame_skip=self.FRAME_SKIP)
        self.policy_net = QLearning(Experiment.ctx,
                                    Experiment.DISCOUNT,
                                    Experiment.INPUT_SAMPLE)
        self.player = Player(self.game, self.policy_net, Experiment.rng)
        self.replay_buffer = ReplayBuffer(Experiment.WIDTH,
                                          Experiment.HEIGHT,
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
        while steps_left > 0:
            ep_steps, ep_reward = self.player.run_episode(epoch, steps_left, self.replay_buffer, True)
            steps_left -= ep_steps
        print('###########################  epoch [%d] done' % epoch)


if __name__ == '__main__':
    exper = Experiment()
    exper.start()

    pass
