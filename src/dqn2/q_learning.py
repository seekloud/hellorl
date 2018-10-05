# Author: Taoz
# Date  : 8/26/2018
# Time  : 2:47 PM
# FileName: q_network.py

from mxnet import autograd, gluon

import src.utils as g_utils
from src.dqn2.network import *

from src.dqn2.config import *
import time
import numpy as np

from src.ztutils import CirceBuffer


class QLearning(object):

    def __init__(self, ctx, model_file=None):
        self.ctx = ctx
        self.update_count = 0
        self.train_count = 0

        self.time_statistic = CirceBuffer(300)
        self.target_net_update_interval = TARGET_NET_UPDATE_INTERVAL

        self.policy_net = get_net(ACTION_NUM, ctx=self.ctx)
        self.target_net = get_net(ACTION_NUM, ctx=self.ctx)

        if model_file is not None:
            print('%s: QLearning read trained model from [%s]' % (time.strftime("%Y-%m-%d %H:%M:%S"), model_file))
            self.policy_net.load_parameters(model_file, ctx=self.ctx)

        self._update_target_net()

        self.trainer = gluon.Trainer(self.policy_net.collect_params(), OPTIMIZER,
                                     {'learning_rate': LEARNING_RATE,
                                      'wd': WEIGHT_DECAY})
        self.loss_func = gluon.loss.L2Loss()

    def _update_target_net(self):
        self.update_count += 1
        print('QLearning update_target_net[%d] at train[%d]' % (self.update_count, self.train_count))
        copy_parameters(self.policy_net, self.target_net)

    def train_policy_net(self,
                         batch_size: int,
                         images: np.ndarray,
                         actions: np.ndarray,
                         rewards: np.ndarray,
                         terminals: np.ndarray):

        """
        Train one batch.

        Arguments:

        imgs - b x (f + 1) x C x H x W numpy array, where b is batch size,
               f is num frames, h is height and w is width.
        actions - b ndarray array of integers
        rewards - b ndarray array
        terminals - b numpy boolean array (currently ignored)

        Returns: average loss
        """
        t0 = time.time()
        # print('image shape=', images.shape, images.dtype)
        # print('actions shape=', actions.shape, actions.dtype)
        # print('rewards shape=', rewards.shape, rewards.dtype)
        # print('terminals shape=', terminals.shape, terminals.dtype)

        images = nd.array(images, dtype='float32', ctx=self.ctx) / 255.0

        st = images[:, :-1]
        s = st.shape
        st = st.reshape((s[0], -1, s[-2], s[-1]))  # batch x (f x C) x H x W

        at = nd.array(actions, dtype='uint8', ctx=self.ctx)
        rt = nd.array(rewards, dtype='float32', ctx=self.ctx)
        tt = nd.array(terminals, dtype='float32', ctx=self.ctx)
        st1 = images[:, 1:].reshape((s[0], -1, s[-2], s[-1]))  # batch x (f x C) x H x W

        self.train_count += 1

        # print('st shape=', st.shape, st.dtype)
        # print('at shape=', at.shape, at.dtype)
        # print('rt shape=', rt.shape, rt.dtype)
        # print('tt shape=', tt.shape, tt.dtype)
        # print('st1 shape=', st1.shape, st1.dtype)
        # print(' - - ')

        next_qs = self.target_net(st1)
        next_q_out = nd.max(next_qs, axis=1)
        target = rt + next_q_out * (1.0 - tt) * DISCOUNT

        # _print_info('next_qs', next_qs)
        # _print_info('next_q_out', next_q_out)
        # _print_info('target', target)
        # print('- -')

        with autograd.record():
            current_qs = self.policy_net(st)
            # _print_info('current_qs', current_qs)
            current_q = nd.pick(current_qs, at, 1)
            # _print_info('current_q', current_q)
            loss = self.loss_func(target, current_q)
            # _print_info('loss', loss)

        loss.backward()

        # 梯度裁剪
        if GRAD_CLIPPING_THETA is not None:
            params = [p.data() for p in self.policy_net.collect_params().values()]
            g_utils.grad_clipping(params, GRAD_CLIPPING_THETA, self.ctx)

        self.trainer.step(batch_size)

        if (self.train_count + 1) % self.target_net_update_interval == 0:
            self._update_target_net()

        # total_loss = 0.0
        total_loss = loss.mean().asscalar()
        t1 = time.time()

        self.time_statistic.add(t1 - t0)

        if (self.train_count + 1) % 100 == 0:
            print('\n[%s] Train [%d] finish. avg_train_time: %.3f' %
                  (time.strftime("%Y-%m-%d %H:%M:%S"), self.train_count, self.time_statistic.avg()))

        return total_loss


def _print_info(name, data):
    print('name=[%s] [%s, %s]' % (name, str(data.shape), str(data.dtype)))
