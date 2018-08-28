# Author: Taoz
# Date  : 8/26/2018
# Time  : 2:47 PM
# FileName: q_network.py

import numpy as np
import mxnet as mx
from mxnet import init, nd, autograd, gluon
from mxnet.gluon import data as gdata, nn, loss as gloss


class QLearning(object):

    def __init__(self, ctx, discount=0.99, params_path=None):
        self.ctx = ctx
        self.discount = discount
        self.policy_net = self.get_net(18)
        self.target_net = self.get_net(18)
        self.update_target_net()

        learning_rate = 0.05
        weight_decay = 0

        self.trainer = gluon.Trainer(self.policy_net.collect_params(), 'adam',
                                {'learning_rate': learning_rate,
                                 'wd': weight_decay})


    def update_target_net(self):
        p_params = self.policy_net.collect_params()
        t_params = self.target_net.collect_params()
        t_params.update(p_params)

    def choose_action(self, state):
        return np.random.randint(0, 18)

    def train_policy_net(self, imgs, actions, rs, terminals):
        """
        Train one batch.

        Arguments:

        imgs - b x (f + 1) x C x H x H numpy array, where b is batch size,
               f is num frames, h is height and w is width.
        actions - b x 1 numpy array of integers
        rewards - b x 1 numpy array
        terminals - b x 1 numpy boolean array (currently ignored)

        Returns: average loss
        """
        batch_size = actions.shape[0]

        states = imgs[:, :-1, :, :, :]
        next_states = imgs[:, 1:, :, :, :]
        s = states.shape
        states = states.reshape((s[0], -1, s[3], s[4]))  # batch x (f x C) x H x H
        next_states = next_states.reshape((s[0], -1, s[3], s[4]))  # batch x (f x C) x H x H

        st = nd.array(states, ctx=self.ctx, dtype=np.float32) / 255.0
        at = nd.array(actions[:, 0], ctx=self.ctx)
        rt = nd.array(rs[:, 0], ctx=self.ctx)
        tt = nd.array(terminals[:, 0], ctx=self.ctx)
        st1 = nd.array(next_states, ctx=self.ctx, dtype=np.float32) / 255.0

        next_qs = self.target_net(st1)
        next_q_out = nd.max(next_qs, axis=1)
        target = rt + next_q_out * (1.0 - tt) * self.discount

        with autograd.record():
            current_qs = self.policy_net(st)
            current_q = nd.choose_element_0index(current_qs, at)
            loss = nd.clip(target - current_q, -100, 100)
            total_loss = nd.sum(nd.abs(loss))
        total_loss.backward()
        self.trainer.step(batch_size)

        return total_loss.asnumpy()

    def copy_params_from(self, other_net):
        pass

    def q_vals(self, sample_batch):
        pass

    def save_params_to_file(self, path):
        pass

    def get_net(self, action_num):
        net = nn.Sequential()
        with net.name_scope():
            net.add(
                nn.Conv2D(channels=32, kernel_size=8, strides=4, padding=4, activation='relu'),
                nn.Conv2D(channels=64, kernel_size=4, strides=2, padding=2, activation='relu'),
                nn.Conv2D(channels=64, kernel_size=3, strides=1, padding=1, activation='relu'),
                nn.Flatten(),
                nn.Dense(512, activation="relu"),
                nn.Dense(action_num)
            )
        net.initialize(init.Xavier(), ctx=self.ctx)
        return net

transform_test = gdata.vision.transforms.Compose([
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                      [0.2023, 0.1994, 0.2010])
])


def getNet():
    pass
