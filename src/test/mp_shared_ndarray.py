# Author: Taoz
# Date  : 9/24/2018
# Time  : 7:15 PM
# FileName: mp_shared_ndarray.py


import multiprocessing as mp
import numpy as np
import ctypes
import time


# N, M = 100, 11
# shared_arr = mp.Array(ctypes.c_double, N)


def error_handle(name):
    def func(e: Exception):
        print('!!!!!!!!!!!    [%s] error happen: %s, %s' % (name, str(e), str(e.__cause__)))

    return func


def cacl_size(shape: tuple, dtype: str):
    bytes_len = 0
    if dtype == 'int32' or \
            dtype == 'int' or \
            dtype == 'float' or \
            dtype == 'float32' or \
            dtype == 'float':
        bytes_len = 4
    elif dtype == 'int8' or dtype == 'uint8':
        bytes_len = 1
    elif dtype == 'float64' or dtype == 'int64':
        bytes_len = 8
    else:
        raise Exception("error dtype:", dtype)

    length = 1
    for x in shape:
        length *= x

    size = bytes_len * length
    return size


def reader1(shared_arr, shape, dtype, d):
    print('reader start.')
    size = cacl_size(shape, dtype)
    np_arr: np.ndarray = tonumpyarray(shared_arr, shape, dtype)

    print('in reader, waiting...')
    time.sleep(5.0 * d + 5)
    np_arr[0, 0, 0] += 1.0
    print('in reader, size:', size)
    print('in reader, x=', np_arr[0, 0, 0])


def test5():
    shape = (2, 5)
    dtype = 'uint8'

    start_method = 'spawn'

    size = cacl_size(shape, dtype)
    mp_ctx = mp.get_context(start_method)

    shared_arr = mp_ctx.Array(ctypes.c_byte, size)
    np_arr: np.ndarray = tonumpyarray(shared_arr, shape, dtype)
    np_arr[0, 0] = 999

    print('-')
    print(type(shared_arr.get_obj()))
    print(shared_arr.get_obj())

    print(np_arr)

    np_arr2: np.ndarray = tonumpyarray(shared_arr, shape, dtype)
    print(np_arr2)


def test4():
    shape = (2, 5)
    dtype = 'float32'

    start_method = 'spawn'

    size = cacl_size(shape, dtype)
    mp_ctx = mp.get_context(start_method)

    shared_arr = mp_ctx.Array(ctypes.c_double, size)
    np_arr: np.ndarray = tonumpyarray(shared_arr, shape, dtype)
    print('-')
    print(type(shared_arr.get_obj()))
    print(shared_arr.get_obj())

    print(np_arr)


def test3():
    shape = (300, 400, 5000)
    dtype = 'float32'

    start_method = 'forkserver'

    size = cacl_size(shape, dtype)
    mp_ctx = mp.get_context('forkserver')

    shared_arr = mp_ctx.Array(ctypes.c_double, size)
    np_arr: np.ndarray = tonumpyarray(shared_arr, shape, dtype)

    ps = []
    for i in range(20):
        p = mp_ctx.Process(target=reader1, args=(shared_arr, shape, dtype, i))
        ps.append(p)
        p.start()

    for i in range(10):
        print('in main waiting...')
        time.sleep(1.0)
        print('in main, set')
        np_arr[0, 0, 0] = 99.99999
        print('in main, size:', size)
        print('in main, x=', np_arr[0, 0, 0])

    for p in ps:
        p.join()
    print('DONE')

    pass


def test2():
    shape = (3, 4, 5, 6)
    dtype = 'float32'
    size = cacl_size(shape, dtype)
    print(size)

    shared_arr = mp.Array(ctypes.c_double, size)

    np_arr: np.ndarray = tonumpyarray(shared_arr, shape, dtype)
    print(np_arr)
    print(np_arr.shape)
    print(np_arr.dtype)


def test1():
    N = 10
    shared_arr = mp.Array(ctypes.c_double, N)

    np_arr: np.ndarray = tonumpyarray(shared_arr, (1, N), np.float32)
    print(np_arr.dtype)
    print(np_arr.shape)
    print(np_arr)

    pass


def tonumpyarray(mp_arr, shape: tuple, dtype=np.float32):
    return np.frombuffer(mp_arr.get_obj(), dtype=dtype).reshape(shape)


if __name__ == '__main__':
    # mp.freeze_support()
    test5()
    pass
