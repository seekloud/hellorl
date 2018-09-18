# Author: Taoz
# Date  : 9/16/2018
# Time  : 10:56 PM
# FileName: player.py

import multiprocessing as mp
import queue


import random
import time

def start_player(play_id: int,
                 observation_queue: queue.Queue,
                 action_in: mp.Connection,
                 experience_queue: queue.Queue
                 ):
    # create player and start it.
    print('in _start_player', play_id, observation_queue, action_in)
    for i in range(10):
        print('[%d] %d' % (play_id, i))
        t = random.randint(10, 100) / 1000.1
        time.sleep(t)
    pass


class Player(object):
    def __init__(self,
                 name,
                 observation_queue: queue.Queue,
                 action_in: mp.Pipe):
        self.env = None

        pass

    def run_episode(self):
        pass