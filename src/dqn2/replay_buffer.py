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
                 capacity: int):

        self.width = width
        self.height = height
        self.channel = channel
        self.capacity = capacity
        self.phi_length = phi_length
        self.discount = discount
        self.rng = rng

        self.images = np.zeros((capacity, channel, height, width), dtype='uint8')
        self.actions = np.zeros((capacity,), dtype='uint8')
        self.rewards = np.zeros((capacity,), dtype='float32')
        self.terminals = np.zeros((capacity,), dtype='int8')

        self.size = 0
        self.top = 0

    def add_experience(self, length, images: list, actions: list, rewards: list):

        assert length == len(images)
        assert length == len(actions)
        assert length == len(rewards)
        assert length < (self.capacity / 2.0)

        self.size = min(self.capacity, self.size + length)
        back_capacity = self.capacity - self.top

        terminals_0 = np.zeros(length, dtype='int8').reshape((length,))
        terminals_0[-1] = 1
        terminals = terminals_0.tolist()

        if length <= back_capacity:
            self.images[self.top: (self.top + length)] = images
            self.actions[self.top: (self.top + length)] = actions
            self.rewards[self.top: (self.top + length)] = rewards
            self.terminals[self.top: (self.top + length)] = terminals
            if length == back_capacity:
                self.top = 0
            else:
                self.top = self.top + length
        else:
            self.images[self.top: self.capacity] = images[:back_capacity]
            self.actions[self.top: self.capacity] = actions[:back_capacity]
            self.rewards[self.top: self.capacity] = rewards[:back_capacity]
            self.terminals[self.top: self.capacity] = terminals[:back_capacity]
            rest = length - back_capacity
            self.images[:rest] = images[back_capacity:]
            self.actions[:rest] = actions[back_capacity:]
            self.rewards[:rest] = rewards[back_capacity:]
            self.terminals[:rest] = terminals[back_capacity:]
            self.top = rest

    def random_batch(self, batch_size):

        begin = 0

        end = self.capacity
        if self.size < self.capacity:
            end = self.top
        end = end - self.phi_length

        indices = np.random.uniform(begin, end, (batch_size,)).astype('int32')

        images = np.zeros((batch_size,
                           self.phi_length + 1,
                           self.channel,
                           self.height,
                           self.width),
                          dtype='uint8')


        for i in range(batch_size):
            target_begin = indices[i]
            target_end = target_begin + self.phi_length + 1
            # get (phi_length + 1) images
            images[i] = self.images[target_begin: target_end]

        actions = np.take(self.actions, indices, axis=0)
        rewards = np.take(self.rewards, indices, axis=0)
        terminals = np.take(self.terminals, indices, axis=0)

        # actions = nd.take(self.actions, indices)
        # rewards = nd.take(self.rewards, indices)
        # terminals = nd.take(self.terminals, indices)

        return images, actions, rewards, terminals


def test_add():
    height = 10
    width = 5
    channel = 2
    phi_length = 4
    discount = 0.99
    capacity = 27
    rng = np.random.RandomState(100)

    buffer = ReplayBuffer(height, width, channel, phi_length, discount, rng, capacity)

    def get_experience(rows, begin=100):
        images = np.arange(begin, begin + (rows * channel * height * width)).reshape((rows, channel, height, width))
        actions = np.arange(begin, begin + rows)
        rewards = np.arange(begin, begin + rows)
        return rows, images.tolist(), actions.tolist(), rewards.tolist()

    buffer.add_experience(*get_experience(10))
    buffer.add_experience(*get_experience(10))
    buffer.add_experience(*get_experience(8))

    print('image:\n', buffer.images)
    print('actions:\n', buffer.actions)
    print('rewards:\n', buffer.rewards)
    print('terminals:\n', buffer.terminals)

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

    buffer = ReplayBuffer(height, width, channel, phi_length, discount, rng, capacity)

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

    images, actions, rewards, terminals = data

    print('-------------------')
    print(images)
    print(actions)
    print(rewards)
    print(terminals)
    print('DONE.')

    pass


if __name__ == '__main__':
    test_get()
