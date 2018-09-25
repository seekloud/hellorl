# Author: Taoz
# Date  : 7/1/2018
# Time  : 11:26 PM
# FileName: ztutils.py
import os
import numpy as np


def mkdir_if_not_exist(path):
    print('mkdir_if_not_exist:[%s]' % path)
    if not os.path.exists(path):
        os.makedirs(path)


class CirceBuffer(object):
    def __init__(self, capacity: int):
        assert capacity > 0
        self._capacity = capacity
        self._list = []
        self._begin = 0
        self._sum = 0.0

    def add(self, num: float):
        self._sum += num
        if self.size() < self._capacity:
            self._list.append(num)
        else:
            self._sum -= self._list[self._begin]
            self._list[self._begin] = num
            self._begin = (self._begin + 1) % self._capacity

    def avg(self):
        length = self.size()
        if length > 0:
            return self._sum / length
        else:
            return 0.0

    def size(self):
        return len(self._list)

    def clean(self):
        self._begin = 0
        self._sum = 0.0
        self._list = []


def tonumpyarray(mp_arr, shape: tuple, dtype=np.float32):
    return np.frombuffer(mp_arr.get_obj(), dtype=dtype).reshape(shape)
