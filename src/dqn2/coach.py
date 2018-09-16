# Author: Taoz
# Date  : 9/16/2018
# Time  : 10:57 PM
# FileName: coach.py

import queue
import multiprocessing as mp
import multiprocessing.managers.Queue as q


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
