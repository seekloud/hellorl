# Author: Taoz
# Date  : 9/6/2018
# Time  : 11:09 AM
# FileName: test1.py


import multiprocessing as mp


def start_worker(id, v):
    print('i am worker %d' % id)
    print('value is %d' % v.value)
    pass


def test_process():
    mp_ctx = mp.get_context('forkserver')

    v = mp_ctx.Value('i', 10)
    # mp_ctx = mp.get_context('spawn')
    a = 123
    p = mp_ctx.Process(target=start_worker, args=(a, v))
    p.start()
    p.join()
    print('p.start.')
    print('DONE.')


if __name__ == '__main__':
    test_process()
