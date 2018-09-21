# Author: Taoz
# Date  : 9/16/2018
# Time  : 10:57 PM
# FileName: coach.py

import queue
import multiprocessing as mp
from src.dqn2.network import *
from src.dqn2.q_learning import QLearning
from src.dqn2.replay_buffer import ReplayBuffer
import src.utils as utils
import os


def start_coach(pre_trained_model_file: str,
                experience_queue: queue.Queue,
                play_net_version
                ):
    # create coach, and start it.
    coach = Coach(pre_trained_model_file, experience_queue, play_net_version)
    print('+++++++++++++++++++ Coach begin.')
    coach.start()
    print('Coach finish.')


class Coach(object):

    def __init__(self,
                 model_file: str,
                 experience_queue: queue.Queue,
                 shared_play_net_version):
        self.ctx = utils.try_gpu(GPU_INDEX)
        self.replay_buffer = ReplayBuffer(HEIGHT, WIDTH, CHANNEL, PHI_LENGTH, DISCOUNT, RANDOM, BUFFER_MAX, self.ctx)

        self.experience_queue = experience_queue
        self.shared_play_net_version = shared_play_net_version
        self.episode_count = 0
        self.step_count = 0
        self.train_count = 0

        self.q_learning = QLearning(self.ctx, model_file)

    def start(self):
        while True:
            self._read_experience()
            self._train()
            if self.train_count % 10 == 0:
                self._update_play_net()

    def _read_experience(self):
        count = 0
        self.experience_queue.put(-1)
        while True:
            experience = self.experience_queue.get()
            if not isinstance(experience, int):
                count += 1
                self._save_experience(experience)
            elif experience == -1:
                break
            else:
                print('error experience_queue code:', experience)
                break
        print('Coach read episode [%d], total step=[%d], total episode=[%d]' %
              (count, self.step_count, self.episode_count))

    def _save_experience(self, experience):
        # experience = (step_count, images, actions, rewards)
        (length, images, actions, rewards) = experience
        self.replay_buffer.add_experience(length, images, actions, rewards)
        self.step_count += length
        self.episode_count += 1
        print('save steps: %d' % length)

    def _train(self):
        self.train_count += 1
        bs = BATCH_SIZE
        images, actions, rewards, terminals = self.replay_buffer.random_batch(bs)
        loss = self.q_learning.train_policy_net(bs, images, actions, rewards, terminals)
        return loss

    def _update_play_net(self):
        file_path = PLAY_NET_MODEL_FILE
        # delete file first?
        # if os.path.exists(file_path):
        #     os.remove(file_path)
        save_model_to_file(self.q_learning.policy_net, file_path)
        self.shared_play_net_version.value = self.train_count


def foo(a: int, qu: queue.Queue):
    print(a)
    print(qu)
    b: int = 5
    print(b)
    pass


if __name__ == '__main__':
    m = mp.Manager()
    q = m.Queue()
    foo(1, q)
