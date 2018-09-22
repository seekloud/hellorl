# Author: Taoz
# Date  : 9/16/2018
# Time  : 10:56 PM
# FileName: replay_buffer.py


import mxnet as mx
from mxnet import nd
import numpy as np
import time


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
        self.actions = nd.zeros((capacity, 1), dtype='uint8', ctx=ctx)
        self.rewards = nd.zeros((capacity, 1), dtype='float32', ctx=ctx)
        self.terminals = nd.zeros((capacity, 1), dtype='int8', ctx=ctx)

        self.size = 0
        self.top = 0

    def add_experience(self, length, images: list, actions: list, rewards: list):

        assert length == len(images)
        assert length == len(actions)
        assert length == len(rewards)
        assert length < (self.capacity / 2.0)

        self.size = min(self.capacity, self.size + length)
        back_capacity = self.capacity - self.top

        images_ = nd.array(images, ctx=self.ctx)
        actions_ = nd.array(actions, ctx=self.ctx).reshape((length, 1))
        rewards_ = nd.array(rewards, ctx=self.ctx).reshape((length, 1))
        terminals_ = nd.zeros(length, dtype='int8', ctx=self.ctx).reshape((length, 1))
        terminals_[-1, 0] = 1

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

        end = self.capacity
        if self.size < self.capacity:
            end = self.top
        end = end - self.phi_length

        # indices = nd.random.uniform(begin, end, (batch_size,), ctx=self.ctx).astype('int32')
        indices_list = np.random.uniform(begin, end, (batch_size,)).astype('int32')
        indices = nd.array(indices_list, ctx=self.ctx).astype('int32')

        print('1111111111111111111111111111')
        images = nd.zeros((batch_size,
                           self.phi_length + 1,
                           self.channel,
                           self.height,
                           self.width),
                          dtype='uint8',
                          ctx=self.ctx)

        print('images shape=', images.shape, images.dtype)

        for i in range(batch_size):
            target_begin = indices_list[i]
            target_end = target_begin + self.phi_length + 1
            # get (phi_length + 1) images
            images[i] = self.images[target_begin: target_end]

        actions = nd.take(self.actions, indices)
        rewards = nd.take(self.rewards, indices)
        terminals = nd.take(self.terminals, indices)

        return images, actions, rewards, terminals


def test_add():
    height = 10
    width = 5
    channel = 2
    phi_length = 4
    discount = 0.99
    capacity = 27
    rng = np.random.RandomState(100)
    ctx = mx.cpu()

    buffer = ReplayBuffer(height, width, channel, phi_length, discount, rng, capacity, ctx)

    def get_experience(rows, begin=100):
        images = np.arange(begin, begin + (rows * channel * height * width)).reshape((rows, channel, height, width))
        actions = np.arange(begin, begin + rows)
        rewards = np.arange(begin, begin + rows)
        return rows, images.tolist(), actions.tolist(), rewards.tolist()

    buffer.add_experience(*get_experience(10))
    buffer.add_experience(*get_experience(10))
    buffer.add_experience(*get_experience(10))

    # print('image:\n', buffer.images)
    # print('actions:\n', buffer.actions)
    # print('rewards:\n', buffer.rewards)
    # print('terminals:\n', buffer.terminals)

    print('capacity=', buffer.capacity)
    print('size=', buffer.size)
    print('top=', buffer.top)


def test_get():
    height = 10
    width = 5
    channel = 2
    phi_length = 4
    discount = 0.99
    capacity = 300
    rng = np.random.RandomState()

    ctx = mx.cpu()
    mx.random.seed(int(time.time() * 1000), ctx)

    buffer = ReplayBuffer(height, width, channel, phi_length, discount, rng, capacity, ctx)

    def get_experience(rows, begin=0):
        images = np.arange(begin, begin + (rows * channel * height * width)).reshape((rows, channel, height, width))
        actions = np.arange(begin, begin + rows) * 1
        rewards = np.arange(begin, begin + rows) * 0.1
        return rows, images.tolist(), actions.tolist(), rewards.tolist()

    buffer.add_experience(*get_experience(130))
    buffer.add_experience(*get_experience(130))
    buffer.add_experience(*get_experience(130))

    print('image:\n', buffer.images)
    print('actions:\n', buffer.actions)
    print('rewards:\n', buffer.rewards)
    print('terminals:\n', buffer.terminals)

    print('capacity=', buffer.capacity)
    print('size=', buffer.size)
    print('top=', buffer.top)

    data = buffer.random_batch(4)

    print('-------------------')
    print(data)
    print('DONE.')

    pass


if __name__ == '__main__':
    test_get()
