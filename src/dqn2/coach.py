# Author: Taoz
# Date  : 9/16/2018
# Time  : 10:57 PM
# FileName: coach.py


import src.utils as utils
from src.dqn2.network import *
from src.dqn2.q_learning import QLearning
from src.dqn2.replay_buffer import ReplayBuffer

import multiprocessing as mp
import threading
import queue


def start_coach(pre_trained_model_file,
                experience_in_list,
                play_net_version
                ):
    # create coach, and start it.
    print('++++++++++++++++++ Coach starting....')
    coach = Coach(pre_trained_model_file, experience_in_list, play_net_version)
    coach.start()
    print('Coach finish.')


def listen_experience(experience_in,
                      merge_queue: queue.Queue,
                      queue_lock: threading.Lock):
    print('Coach listen_experience by: ', threading.current_thread().name)
    while True:
        experience = experience_in.recv()
        with queue_lock:
            merge_queue.put(experience)


class Coach(object):

    def __init__(self,
                 model_file: str,
                 experience_in_list: list,
                 shared_play_net_version):
        self.ctx = utils.try_gpu(GPU_INDEX)
        self.replay_buffer = ReplayBuffer(HEIGHT, WIDTH, CHANNEL, PHI_LENGTH, DISCOUNT, RANDOM, BUFFER_MAX, self.ctx)

        self.experience_in_list = experience_in_list
        self.shared_play_net_version = shared_play_net_version
        self.episode_count = 0
        self.step_count = 0
        self.train_count = 0
        self.local_experience_queue = queue.Queue(200)
        self.queue_lock = threading.Lock()

        self.q_learning = QLearning(self.ctx, model_file)

    def start(self):
        count = 0
        for exp_in in self.experience_in_list:
            t = threading.Thread(target=listen_experience,
                                 args=(exp_in, self.local_experience_queue, self.queue_lock),
                                 name='queue_loader_' + str(count))
            count += 1
            t.start()

        while True:
            self._read_experience()
            self._train()
            if self.train_count + 1 % 2 == 0:
                self._update_play_net()

    # def _read_experience(self):
    #     count = 0
    #     while True:
    #         print(' - - Coach waiting...')
    #         experience = self.experience_queue.get()
    #         print(' - - Coach get data:')
    #         self._save_experience(experience)
    #         print(' - - Coach saved experience.')

    def _read_experience(self):
        qu = self.local_experience_queue
        experience_list = []
        if not qu.empty():
            with self.queue_lock:
                while not qu.empty():
                    experience = qu.get()
                    experience_list.append(experience)
        for exp in experience_list:
            self._save_experience(exp)
        print(' - - - Coach read [%d] episodes, total step=[%d], total episode=[%d]' %
              (len(experience_list), self.step_count, self.episode_count))

    def _save_experience(self, experience):

        (player_id, length, images, actions, rewards) = experience
        print('coach got experience from player[%d], length=%d' % (player_id, length))
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
        # time.sleep(0.1)
        # print(' - Coach skip train in test...')

    def _update_play_net(self):

        # delete file first?
        # if os.path.exists(file_path):
        #     os.remove(file_path)

        file_path = PLAY_NET_MODEL_FILE
        save_model_to_file(self.q_learning.policy_net, file_path)
        self.shared_play_net_version.value = self.train_count
        # time.sleep(0.1)
        # print(' - Coach skip update_play_net..')
