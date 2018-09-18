# Author: Taoz
# Date  : 9/16/2018
# Time  : 10:57 PM
# FileName: coach.py

import queue
import multiprocessing as mp


def start_coach(experience_queue: queue.Queue,
                pre_trained_model_file: str,
                shared_inform
                ):
    # create coach, and start it.
    pass


class Coach(object):

    def __init__(self, experience_queue: queue.Queue):
        pass

    def train(self):
        pass


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
