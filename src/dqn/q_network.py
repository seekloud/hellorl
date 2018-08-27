# Author: Taoz
# Date  : 8/26/2018
# Time  : 2:47 PM
# FileName: q_network.py

import numpy as np


class QNetwork(object):
    def __init__(self, params_path=None):
        self.net = ''
        pass

    def choose_action(self, state):
        return np.random.randint(0, 18)

    def update(self, sample_batch):
        pass

    def copy_params_from(self, other_net):
        pass

    def q_val(self, sample_batch):
        pass

    def save_params_to_file(self, path):
        pass
