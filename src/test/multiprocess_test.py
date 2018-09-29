# Author: Taoz
# Date  : 9/6/2018
# Time  : 11:09 AM
# FileName: test1.py


import multiprocessing as mp
import time
import sys
import random
import mxnet as mx
from mxnet import nd
import numpy as np


def test_queue():
    worker_num = 5
    manager = mp.Manager()
    task_queue = manager.Queue()
    pool = mp.Pool(worker_num)

    # ctx = mx.cpu(0)

    # d0 = nd.array([1, 2, 3], ctx=ctx, dtype=np.float32)
    #
    # print(d0)
    # print(d0.shape)

    for i in range(worker_num):
        pool.apply_async(count, ('WORKER[%d]' % i, task_queue))

    data = [[1, 2, 3, 4], [5, 6, 7, 8]]
    print('src list:', data)
    t0 = time.time()
    for i in range(100):
        task_queue.put(data)
        s = task_queue.qsize()
        t = 0.01
        if s > 2000:
            print('qsize=%d, wait some %f second.' % (s, t))
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
        elif isinstance(data, list):
            print('i[%s] got list %s' % (name, data))
            pass
        else:
            try:
                length = len(data)
                if length > 100000:
                    break
                # print("%s, %d ----- data load %d" % (name, c, len(data)))
            except:
                print("exception happen!!")
    print('i am done. %s i got [%d] data' % (name, c))


def count1(name, in_queue):
    print('hello, i am %s' % name)
    c = 0
    while True:
        data = in_queue.get()
        c += 1
        if data is None:
            break
        elif isinstance(data, list):
            print('i[%s] got list %s' % (name, data))
            pass
        else:
            try:
                length = len(data)
                if length > 100000:
                    break
                # print("%s, %d ----- data load %d" % (name, c, len(data)))
            except:
                print("exception happen!!")
    print('i am done. %s i got [%d] data' % (name, c))


def ping(cnn):
    for i in range(3):
        data = [i for i in range(i)]
        cnn.send(data)
        print('i am ping, i send: ', data)
    cnn.send(None)
    pass


def pong(cnn):
    date = cnn.recv()
    while date is not None:
        print('i am pong, i got: ', date)
        date = cnn.recv()
    pass


def ping_pong(name, cnn, first=False):
    if first:
        print('[%s] send [0]' % name)
        cnn.send(0)

    data = None
    while True:
        try:
            print('ping_pong wait....')
            data = cnn.recv()
            print('got data:', data)
        except:
            print('recv error.')

        print('111111')
        if data is None:
            print('3333333333')
            print('[%s] got None, and i am finish.' % name)
            break
        else:
            print('222222222222')

            print('[%s] got [%s]' % (name, str(data)))
            if data > 10:
                cnn.send(None)
                print('[%s] send None, and i am finish.' % name)
                break
            else:
                print('[%s] send [%d].' % (name, data + 1))
                cnn.send(data + 1)
    print('---------------- out -----------------')


def test_pipe1():
    worker_num = 2
    pool = mp.Pool(worker_num)
    cnn1, cnn2 = mp.Pipe()

    pool.apply_async(ping, args=(cnn1,))
    pool.apply_async(pong, args=(cnn2,))

    pool.close()
    print('waiting.')
    pool.join()
    print('DONE.')
    pass


def test_pipe2():
    worker_num = 2
    pool = mp.Pool(worker_num)
    cnn1, cnn2 = mp.Pipe()

    pool.apply_async(ping_pong, args=('Li Lei', cnn1, True))
    pool.apply_async(ping_pong, args=('Han Meimei', cnn2,))

    pool.close()
    print('waiting.')
    pool.join()
    print('DONE.')
    pass


def test_pipe3():
    worker_num = 2
    pool = mp.Pool(worker_num)
    cnn1, cnn2 = mp.Pipe()

    pool.apply_async(ping_pong, args=('Li Lei', cnn1, True))
    pool.close()

    data = cnn2.recv()
    print('main get data:', data)
    cnn2.send(['i am main, who are you.'])
    cnn2.close()
    # cnn2.send('999')

    print('waiting.')
    pool.join()
    print('DONE.')
    pass


def change_value(d):
    print('in change_value, value =', d['informs'])
    d['informs'] = 'ok.'
    for i in range(1000):
        d['informs'] = 'ok.' + str(i)


def test_shared_mem():
    mm = mp.Manager()
    pool = mp.Pool(1)
    d = mm.dict()
    d['informs'] = 'hello, world.'

    print('go')
    pool.apply_async(change_value, (d,))

    pool.close()
    print('main value = ', d['informs'])
    print('main value1 = ', d.get('aaa', None))

    t0 = time.time()
    for i in range(1000):
        if d['informs'] == 'hahah':
            print('error.')
    t1 = time.time()

    pool.join()
    t2 = time.time()

    print('DONE. t1=%.3f, t2=%.3f' % ((t1 - t0), (t2 - t0)))


import os


def start_worker(id, v):
    print('i am worker %d' % id)
    print('value is %d' % v.value)
    pass


def test_process():
    mp_ctx = mp.get_context('forkserver')

    v = mp_ctx.Value('i', 10)
    # mp_ctx = mp.get_context('spawn')
    launch(mp_ctx, v)
    print('outsided waiting.')
    time.sleep(10)

    print('p.start.')



def launch(mp_ctx, v):
    id = 123
    p = mp_ctx.Process(target=start_worker, args=(id, v), daemon=False)
    p.start()
    print('insided waiting.')
    # p.join()
    # time.sleep(30)
    return p


class Launcher(object):
    def __init__(self):
        print('init...')
        mp_ctx = mp.get_context('forkserver')
        v = mp_ctx.Value('i', 10)
        launch(mp_ctx, v)
        # print('outsided waiting.')
        # time.sleep(30)
        print('init done.')








def test_shared():
    print('begin test_shared')
    launcher = Launcher()
    print('out outsided waiting.')
    time.sleep(30)
    print('DONE.')


if __name__ == '__main__':
    test_shared()
