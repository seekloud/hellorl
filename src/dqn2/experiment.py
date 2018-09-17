# Author: Taoz
# Date  : 9/16/2018
# Time  : 10:56 PM
# FileName: experiment.py


import multiprocessing as mp
import time
import sys
import random
import mxnet as mx
from mxnet import nd
import queue
import numpy as np
from src.dqn2.coach import Coach
from src.dqn2.player import Player

from src.dqn2.constants import *


class Experiment(object):

    def __init__(self):
        pass

    def start_train(self):
        worker_num = 5

        current_play_net_file = None

        pool = mp.Pool(worker_num + 1)
        manager = mp.Manager()
        shared_inform_map = manager.dict()
        player_observation_queue = manager.Queue()
        experience_queue = manager.Queue()

        players = ['player_' + str(i) for i in range(1, worker_num + 1)]
        player_pips = dict()

        # create players
        for player_name in players:
            action_in, action_out = mp.Pipe()
            pool.apply_async(self._start_player, (player_name, player_observation_queue, action_in))

        pool.apply_async(self._start_coach, (shared_inform_map, player_observation_queue))
        pass

    def _start_player(self, play_name, observation_queue: queue.Queue, action_in: mp.Connection):
        # create player and start it.
        print('in _start_player', play_name, observation_queue, action_in)
        for i in range(10):
            print('%s %d' % (play_name, i))
            t = random.randint(10, 100) / 1000.1
            time.sleep(t)
        pass

    def _start_coach(self, shared_map, observation_queue: queue.Queue):
        # create coach, and start it.
        pass

    def start_test(self):
        pass


def change_value(d):
    print('in change_value, value =', d['informs'])
    d['informs'] = 'ok.'


if __name__ == '__main__':
    exp = Experiment()
    exp.start_train()

    # informs = mp.Value('str', 'hello, world.')
    # mm = mp.Manager()
    # pool = mp.Pool(1)
    # d = mm.dict()
    # d['informs'] = 'hello, world.'
    #
    # print('go')
    # pool.apply_async(change_value, (d,))
    #
    # pool.close()
    # pool.join()
    # print('main value = ', d['informs'])
    # print('main value1 = ', d.get('aaa', None))

    # t0 = time.time()
    # for i in range(1000):
    #     if d['informs'] == 'hahah':
    #         print('error.')
    # t1 = time.time()
    #
    #
    # print('DONE. t=', (t1 - t0))

    pass
