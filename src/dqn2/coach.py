# Author: Taoz
# Date  : 9/16/2018
# Time  : 10:57 PM
# FileName: coach.py


import queue
import threading

import src.utils as utils
from src.dqn2.network import *
from src.dqn2.q_learning import QLearning
from src.dqn2.replay_buffer import ReplayBuffer
import os


def start_coach(pre_trained_model_file,
                experience_in_list,
                play_net_file,
                play_net_version
                ):
    # create coach, and start it.
    pid = os.getpid()
    ppid = os.getppid()
    print('++++++++++++++++++ Coach starting.... pid=[%s] ppid=[%s]' % (str(pid), str(ppid)))
    coach = Coach(pre_trained_model_file, experience_in_list, play_net_file, play_net_version)
    coach.start()
    print('Coach finish.')


def listen_experience(experience_in,
                      merge_queue: queue.Queue):
    print('Coach listen_experience by: ', threading.current_thread().name)
    while True:
        experience = experience_in.recv()
        merge_queue.put(experience)


class Coach(object):

    def __init__(self,
                 model_file: str,
                 experience_in_list: list,
                 shared_play_net_file,
                 shared_play_net_version):
        self.ctx = utils.try_gpu(GPU_INDEX)
        self.replay_buffer = ReplayBuffer(HEIGHT, WIDTH, CHANNEL, PHI_LENGTH, DISCOUNT, RANDOM, BUFFER_MAX)

        self.last_data_time = time.time() + 100.0

        self.experience_in_list = experience_in_list
        self.shared_play_net_version = shared_play_net_version
        self.shared_play_net_file = shared_play_net_file
        self.episode_count = 0
        self.step_count = 0
        self.train_count = 0
        self.local_experience_queue = queue.Queue(200)

        self.q_learning = QLearning(self.ctx, model_file)

        # start listener.
        count = 0
        for exp_in in self.experience_in_list:
            t = threading.Thread(target=listen_experience,
                                 args=(exp_in, self.local_experience_queue),
                                 name='queue_loader_' + str(count),
                                 daemon=False)
            t.start()
            count += 1

    def start(self):
        while True:
            self._read_experience()
            self._train()

            if (self.train_count + 1) % PLAY_NET_UPDATE_INTERVAL == 0:
                self._update_play_net()

            if (self.train_count + 1) % POLICY_NET_SAVE_INTERVAL == 0:
                self._save_policy_net()

            if time.time() - self.last_data_time > 100.0:
                print('Coach no data timeout: %.3f' % (time.time() - self.last_data_time))
                break
        print('[ WARNING ] ----------------------- !!!!!!!!!! Coach stop')

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
        timeout = 1.0
        experience = None
        if not qu.empty():
            try:
                experience = qu.get(timeout=timeout)
            except TimeoutError:
                experience = None
                print('[ WARNING ] read experience queue timeout.')

        if experience is not None:
            self.last_data_time = time.time()
            (player_id, length, images, actions, rewards) = experience
            print('Coach got exp from player[%d], length=%d' % (player_id, length))
            self.replay_buffer.add_experience(length, images, actions, rewards)
            self.step_count += length
            self.episode_count += 1

            if (self.episode_count + 1) % 100 == 0:
                print('Coach total step=[%d], total episode=[%d]' % (self.step_count, self.episode_count))

    def _train(self):
        if self.step_count > 100:
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

        file_path = self.shared_play_net_file
        save_model_to_file(self.q_learning.policy_net, file_path)
        self.shared_play_net_version.value = self.train_count
        # time.sleep(0.1)
        # print(' - Coach skip update_play_net..')

    def _save_policy_net(self):
        current_time = time.strftime("%Y%m%d_%H%M%S")
        net = self.q_learning.policy_net
        file_path = MODEL_PATH + '/' + FILE_PREFIX + "_" + GAME_NAME + "_" + BEGIN_TIME + "_" + current_time + '.model'
        save_model_to_file(net, file_path)
        return file_path

