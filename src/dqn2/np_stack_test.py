# Author: Taoz
# Date  : 9/18/2018
# Time  : 11:11 AM
# FileName: np_stack_test.py





import time
import numpy as np

def test():
    a = np.array(range(30000)).reshape((1, 30000))

    b = np.array(range(30000))

    c = np.row_stack((a, b))

    t0 = time.time()
    for i in range(1000):
        c = np.row_stack((c, b))
    t1 = time.time()

    print('size:', c.shape)
    print('Time:', (t1 - t0))


    pass




if __name__ == '__main__':
    test()
