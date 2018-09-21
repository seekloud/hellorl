# Author: Taoz
# Date  : 8/26/2018
# Time  : 2:47 PM
# FileName: q_network.py

from mxnet import autograd, gluon

import src.utils as g_utils
from src.dqn2.network import *


class QLearning(object):
    def __init__(self, ctx, model_file=None):
        self.ctx = ctx
        self.policy_net = get_net(ACTION_NUM, ctx=self.ctx)
        self.target_net = get_net(ACTION_NUM, ctx=self.ctx)

        if model_file is not None:
            print('%s: read trained model from [%s]' % (time.strftime("%Y-%m-%d %H:%M:%S"), model_file))
            self.policy_net.load_parameters(model_file, ctx=self.ctx)

        self._update_target_net()

        self.trainer = gluon.Trainer(self.policy_net.collect_params(), OPTIMIZER,
                                     {'learning_rate': LEARNING_RATE,
                                      'wd': WEIGHT_DECAY})

        self.loss_func = gluon.loss.L2Loss()

    def _update_target_net(self):
        copy_parameters(self.policy_net, self.target_net)

    def train_policy_net(self,
                         batch_size: int,
                         images: nd.NDArray,
                         actions: nd.NDArray,
                         rewards: nd.NDArray,
                         terminals: nd.NDArray):

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

        print('image shape=', images.shape)
        print('actions shape=', actions.shape)
        print('rewards shape=', rewards.shape)
        print('terminals shape=', terminals.shape)

        # states = images[:, :-1, :, :, :]
        # next_states = images[:, 1:, :, :, :]
        # s = states.shape
        # states = states.reshape((s[0], -1, s[-2], s[-1]))  # batch x (f x C) x H x W
        # next_states = next_states.reshape((s[0], -1, s[-2], s[-1]))  # batch x (f x C) x H x W
        #

        st = images[:, :-1]
        s = st.shape
        st = st.reshape((s[0], -1, s[-2], s[-1]))  # batch x (f x C) x H x W
        at = actions
        rt = rewards
        tt = terminals
        st1 = images[:, 1:].reshape((s[0], -1, s[-2], s[-1]))  # batch x (f x C) x H x W

        print('-----------------------------')
        print('st shape=', st.shape)
        print('at shape=', at.shape)
        print('rt shape=', rt.shape)
        print('tt shape=', tt.shape)

        next_qs = self.target_net(st1)
        next_q_out = nd.max(next_qs, axis=1)
        target = rt + next_q_out * (1.0 - tt) * DISCOUNT

        with autograd.record():
            current_qs = self.policy_net(st)
            current_q = nd.pick(current_qs, at, 1)
            loss = self.loss_func(target, current_q)

        loss.backward()

        # 梯度裁剪
        if GRAD_CLIPPING_THETA is not None:
            params = [p.data() for p in self.policy_net.collect_params().values()]
            g_utils.grad_clipping(params, GRAD_CLIPPING_THETA, self.ctx)

        self.trainer.step(batch_size)
        total_loss = loss.mean().asscalar()
        return total_loss
