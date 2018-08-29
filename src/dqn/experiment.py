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
import src.ztutils as ztutils

from src.dqn.config import *


class Experiment(object):
    ctx = utils.try_gpu(0)

    INPUT_SAMPLE = nd.random.uniform(0, 255, (1, PHI_LENGTH * CHANNEL, HEIGHT, WIDTH), ctx=ctx) / 255.0

    # print('input:', type(INPUT_SAMPLE))
    # print('input:', INPUT_SAMPLE)

    mx.random.seed(RANDOM_SEED)
    rng = np.random.RandomState(RANDOM_SEED)

    def __init__(self):
        ztutils.mkdir_if_not_exist(MODEL_PATH)
        self.step_count = 0
        self.episode_count = 0
        self.q_learning = QLearning(Experiment.ctx,
                                    Experiment.INPUT_SAMPLE,
                                    DISCOUNT,
                                    model_file=PRE_TRAIN_MODEL_FILE
                                    )
        self.game = GameEnv(game=GAME_NAME,
                            obs_type=OBSERVATION_TYPE,
                            frame_skip=FRAME_SKIP)
        self.player = Player(self.game,
                             self.q_learning,
                             Experiment.rng)

        self.replay_buffer = ReplayBuffer(HEIGHT,
                                          WIDTH,
                                          CHANNEL,
                                          Experiment.rng,
                                          DISCOUNT,
                                          BUFFER_MAX)

    def start_train(self):
        for i in range(1, EPOCH_NUM + 1):
            self._run_epoch(i)
        print('train done.')
        self.game.close()

    def start_test(self):
        assert PRE_TRAIN_MODEL_FILE is not None
        for i in range(1, EPOCH_NUM + 1):
            self._run_epoch(i, testing=True, render=True)
        print('test done.')
        self.game.close()

    def _run_epoch(self, epoch, testing=False, render=False):
        steps_left = EPOCH_LENGTH
        random_action = True
        last_update_target_step = 0
        while steps_left > 0:
            if self.step_count > BEGIN_RANDOM_STEP:
                random_action = False
            t0 = time.time()
            ep_steps, ep_reward, avg_loss = self.player.run_episode(epoch, steps_left, self.replay_buffer, render=render,
                                                          random_action=random_action, testing=testing)
            t1 = time.time()
            self.step_count += ep_steps
            self.episode_count += 1
            steps_left -= ep_steps
            print('++++++ episode [%d] finish, episode step=%d, total_step=%d, time=%f s, ep_reward=%d, avg_loss=%f'
                  % (self.episode_count, ep_steps, self.step_count, (t1 - t0), int(ep_reward), avg_loss))
            print('')

            if not testing and self.step_count - last_update_target_step > UPDATE_TARGET_PER_STEP and not random_action:
                last_update_target_step = self.step_count
                print('-- -- update_target_net total_step=%d' % self.step_count)
                self.q_learning.update_target_net()

        print('#####  epoch [%d] finish, episode=%d, step=%d \n\n' % (epoch, self.episode_count, self.step_count))
        if not testing:
            self.q_learning.save_params_to_file(MODEL_PATH, 'test1_' + BEGIN_TIME)


def train():
    print(' ====================== START TRAIN ========================')
    exper = Experiment()
    exper.start_train()

def test():
    print(' ====================== START test ========================')
    exper = Experiment()
    exper.start_test()

if __name__ == '__main__':
    train()
