# Author: Taoz
# Date  : 9/16/2018
# Time  : 10:56 PM
# FileName: replay_buffer.py


import mxnet as mx
from mxnet import nd
import numpy as np


class ReplayBuffer(object):
    def __init__(self,
                 height: int,
                 width: int,
                 channel: int,
                 phi_length: int,
                 discount: float,
                 rng: np.random.RandomState,
                 capacity: int,
                 ctx: mx.Context):
        self.width = width
        self.height = height
        self.channel = channel
        self.capacity = capacity
        self.phi_length = phi_length
        self.discount = discount
        self.rng = rng
        self.ctx = ctx

        self.images = nd.zeros((capacity, channel, height, width), dtype='uint8', ctx=ctx)
        self.actions = nd.zeros(capacity, dtype='uint8', ctx=ctx)
        self.rewards = nd.zeros(capacity, dtype='float32', ctx=ctx)
        self.terminals = nd.zeros(capacity, dtype='int8', ctx=ctx)
        self.terminals[-1] = 1  # set terminal for the first episode.

        self.size = 0
        self.top = 0

    def add_experience(self, length, images: list, actions: list, rewards: list):

        assert length == len(images)
        assert length == len(actions)
        assert length == len(rewards)
        assert length < (self.capacity // 2)

        self.size = min(self.capacity, self.size + length)
        back_capacity = self.capacity - self.top

        images_ = nd.array(images, ctx=self.ctx)
        actions_ = nd.array(actions, ctx=self.ctx)
        rewards_ = nd.array(rewards, ctx=self.ctx)
        terminals_ = nd.zeros(length, dtype='int8', ctx=self.ctx)
        terminals_[-1] = 1

        if length <= back_capacity:
            self.images[self.top: (self.top + length)] = images_
            self.actions[self.top: (self.top + length)] = actions_
            self.rewards[self.top: (self.top + length)] = rewards_
            self.terminals[self.top: (self.top + length)] = terminals_
            if length == back_capacity:
                self.top = 0
            else:
                self.top = self.top + length
        else:
            self.images[self.top: self.capacity] = images_[:back_capacity]
            self.actions[self.top: self.capacity] = actions_[:back_capacity]
            self.rewards[self.top: self.capacity] = rewards_[:back_capacity]
            self.terminals[self.top: self.capacity] = terminals_[:back_capacity]
            rest = length - back_capacity
            self.images[:rest] = images_[back_capacity:]
            self.actions[:rest] = actions_[back_capacity:]
            self.rewards[:rest] = rewards_[back_capacity:]
            self.terminals[:rest] = terminals_[back_capacity:]
            self.top = rest

    def random_batch(self, batch_size):

        begin = 0

        # include {end} index.
        end = self.capacity
        if self.size < self.capacity:
            end = self.top
        end = end - self.phi_length

        indices = nd.random.uniform(begin, end + 2, (batch_size,)).astype('int32')

        images = nd.zeros((batch_size,
                           self.phi_length + 1,
                           self.channel,
                           self.height,
                           self.width),
                          dtype='uint8',
                          ctx=self.ctx)

        indices_list = indices.asnumpy().tolist()

        for i in range(batch_size):
            target_begin = indices_list[i]
            target_end = target_begin + self.phi_length + 1
            images[i] = self.images[target_begin: target_end]

        actions = nd.take(self.actions, indices)
        rewards = nd.take(self.rewards, indices)
        terminals = nd.take(self.terminals, indices)

        return images, actions, rewards, terminals


if __name__ == '__main__':
    a = nd.random.uniform(0, 2, (100,), 'float16')
    b = a.astype('int32')
    print(a)
    print(b)
