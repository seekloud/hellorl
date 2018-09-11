# Author: Taoz
# Date  : 9/6/2018
# Time  : 11:09 AM
# FileName: test1.py


import multiprocessing
import time
import sys
import random
import mxnet as mx
from mxnet import nd
import numpy as np


def test():
    worker_num = 5
    manager = multiprocessing.Manager()
    task_queue = manager.Queue()
    pool = multiprocessing.Pool(worker_num)

    # ctx = mx.cpu(0)

    # d0 = nd.array([1, 2, 3], ctx=ctx, dtype=np.float32)
    #
    # print(d0)
    # print(d0.shape)

    for i in range(worker_num):
        pool.apply_async(count, ('WORKER[%d]' % i, task_queue))

    data = [1, 2, 3, 4, 5, 6, 7, 8]
    t0 = time.time()
    for i in range(100000):
        task_queue.put(data)
        s = task_queue.qsize()
        t = 0.01
        if s > 2000:
            print('qsize=%d, wait some %f second.' %(s, t))
            time.sleep(t)

    for i in range(worker_num):
        task_queue.put(None)



    print("go go go.")
    pool.close()
    pool.join()
    t1 = time.time()
    print("ALL FINISH. time=%.4f" % ((t1 - t0) * 1000))




def count(name, in_queue):
    print('hello, i am %s' % name)
    c = 0
    while True:
        data = in_queue.get()
        c += 1
        if data is None:
            break
        else:
            try:
                length = len(data)
                if length > 100000:
                    break
                # print("%s, %d ----- data load %d" % (name, c, len(data)))

            except:
                print("exception happen!!")
    print('i am done. %s i got [%d] data' % (name, c))


if __name__ == '__main__':
    test()
