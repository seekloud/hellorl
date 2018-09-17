# Author: Taoz
# Date  : 9/16/2018
# Time  : 10:56 PM
# FileName: player.py

import multiprocessing as mp
import queue


class Player(object):
    def __init__(self,
                 name,
                 observation_queue: queue.Queue,
                 action_in: mp.Pipe):
        self.env = None

        pass

    def run_episode(self):
        pass