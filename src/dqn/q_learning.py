# Author: Taoz
# Date  : 8/26/2018
# Time  : 2:47 PM
# FileName: q_network.py

import numpy as np
import mxnet as mx
from mxnet import init, nd, autograd, gluon
from mxnet.gluon import data as gdata, nn, loss as gloss
import time
import src.utils as g_utils

clipping_theta = 0.01

from src.dqn.config import DISCOUNT


class QLearning(object):
    def __init__(self, ctx, input_sample, model_file=None):
        self.ctx = ctx
        self.policy_net = self.get_net(18, input_sample)
        self.target_net = self.get_net(18, input_sample)

        if model_file is not None:
            print('%s: read trained model from [%s]' % (time.strftime("%Y-%m-%d %H:%M:%S"), model_file))
            self.policy_net.load_params(model_file, ctx=self.ctx)

        self.update_target_net()

        learning_rate = 0.005
        weight_decay = 0.0

        #adagrad
        self.trainer = gluon.Trainer(self.policy_net.collect_params(), 'adagrad',
                                     {'learning_rate': learning_rate,
                                      'wd': weight_decay})
        self.loss_func = gluon.loss.L2Loss()

    def update_target_net(self):
        copy_params(self.policy_net, self.target_net)

    def choose_action(self, state):
        shape0 = state.shape
        state = nd.array(state, ctx=self.ctx).reshape((1, -1, shape0[-2], shape0[-1]))
        out = self.policy_net(state)
        max_index = nd.argmax(out, axis=1)
        action = max_index.astype(np.uint8).asscalar()
        return action

    def train_policy_net(self, imgs, actions, rs, terminals):
        """
        Train one batch.

        Arguments:

        imgs - b x (f + 1) x C x H x W numpy array, where b is batch size,
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
        states = states.reshape((s[0], -1, s[-2], s[-1]))  # batch x (f x C) x H x W
        next_states = next_states.reshape((s[0], -1, s[-2], s[-1]))  # batch x (f x C) x H x W

        st = nd.array(states, ctx=self.ctx, dtype=np.float32) / 255.0
        at = nd.array(actions[:, 0], ctx=self.ctx)
        rt = nd.array(rs[:, 0], ctx=self.ctx)
        tt = nd.array(terminals[:, 0], ctx=self.ctx)
        st1 = nd.array(next_states, ctx=self.ctx, dtype=np.float32) / 255.0

        next_qs = self.target_net(st1)
        next_q_out = nd.max(next_qs, axis=1)
        target = rt + next_q_out * (1.0 - tt) * DISCOUNT

        with autograd.record():
            current_qs = self.policy_net(st)
            current_q = nd.pick(current_qs, at, 1)
            loss = self.loss_func(target, current_q)
        loss.backward()
        # 梯度裁剪
        params = [p.data() for p in self.policy_net.collect_params().values()]
        g_utils.grad_clipping(params, clipping_theta, self.ctx)

        self.trainer.step(batch_size)
        total_loss = loss.mean().asscalar()
        return total_loss

    def q_vals(self, sample_batch):
        pass

    def save_params_to_file(self, model_path, mark):
        time_mark = time.strftime("%Y%m%d_%H%M%S")
        filename = model_path + '/net_' + str(mark) + '_' + time_mark + '.model'
        self.policy_net.save_params(filename)
        print(time.strftime("%Y-%m-%d %H:%M:%S"), ' save model success:', filename)

    def get_net(self, action_num, input_sample):
        net = nn.Sequential()
        with net.name_scope():
            net.add(
                nn.Conv2D(channels=32, kernel_size=8, strides=4, activation='relu'),
                nn.Conv2D(channels=64, kernel_size=4, strides=2, activation='relu'),
                nn.Conv2D(channels=64, kernel_size=3, strides=1, activation='relu'),
                nn.Flatten(),
                nn.Dense(512, activation="relu"),
                nn.Dense(action_num)
            )
        net.initialize(init.Xavier(), ctx=self.ctx)
        net(input_sample)
        return net


transform_test = gdata.vision.transforms.Compose([
    gdata.vision.transforms.ToTensor(),
    gdata.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                      [0.2023, 0.1994, 0.2010])
])


def copy_params(src_net, dst_net):
    ps_src = src_net.collect_params()
    ps_dst = dst_net.collect_params()
    prefix_length = len(src_net.prefix)
    for k, v in ps_src.items():
        k = k[prefix_length:]
        v_dst = ps_dst.get(k)
        v_dst.set_data(v.data())
