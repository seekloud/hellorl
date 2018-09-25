# Author: Taoz
# Date  : 9/16/2018
# Time  : 10:56 PM
# FileName: replay_buffer.py


import time

import mxnet as mx
import numpy as np
from mxnet import nd

import multiprocessing as mp

from src.dqn2.shared_utils import create_shared_data, to_np_array


def fix_capacity(capacity):
    mod = capacity % 8
    if mod == 0:
        rst = capacity
    else:
        rst = capacity + (8 - mod)
    return rst


def create_replay_buffer_data(height: int,
                              width: int,
                              channel: int,
                              phi_length: int,
                              capacity: int,
                              mp_ctx):
    capacity = fix_capacity(capacity)
    img_shape = (capacity, channel, height, width)
    image_data = create_shared_data(mp_ctx, img_shape, 'uint8')
    action_data = create_shared_data(mp_ctx, (capacity,), dtype='uint8')
    reward_data = create_shared_data(mp_ctx, (capacity,), dtype='float32')
    terminal_data = create_shared_data(mp_ctx, (capacity,), dtype='int8')
    top_value = mp_ctx.Value('i', 0)
    size_value = mp_ctx.Value('i', 0)
    buffer_lock = mp_ctx.Lock()

    return image_data, action_data, reward_data, terminal_data, top_value, size_value, buffer_lock


class ReplayBuffer(object):
    def __init__(self,
                 height: int,
                 width: int,
                 channel: int,
                 phi_length: int,
                 capacity: int,
                 data: tuple):
        capacity = fix_capacity(capacity)

        image_data, action_data, reward_data, terminal_data, top_value, size_value, buffer_lock = data

        self.width = width
        self.height = height
        self.channel = channel
        self.capacity = capacity
        self.phi_length = phi_length
        self.rng = np.random

        img_shape = (capacity, channel, height, width)

        self.images = to_np_array(image_data, img_shape, dtype='uint8')
        self.actions = to_np_array(action_data, (capacity,), dtype='uint8')
        self.rewards = to_np_array(reward_data, (capacity,), dtype='float32')
        self.terminals = to_np_array(terminal_data, (capacity,), dtype='int8')

        self.size_value = size_value
        self.top_value = top_value
        self.lock = buffer_lock

    def add_experience(self, length, images: np.array, actions: list, rewards: list):

        assert length == len(images)
        assert length == len(actions)
        assert length == len(rewards)
        assert length < (self.capacity / 2.0)

        terminals_0 = np.zeros(length, dtype='int8').reshape((length,))
        terminals_0[-1] = 1
        terminals = terminals_0.tolist()

        with self.lock:

            self.size_value.value = min(self.capacity, self.size_value.value + length)

            top = self.top_value.value
            back_capacity = self.capacity - top

            if length <= back_capacity:
                self.images[top: (top + length)] = images
                self.actions[top: (top + length)] = actions
                self.rewards[top: (top + length)] = rewards
                self.terminals[top: (top + length)] = terminals
                if length == back_capacity:
                    self.top_value.value = 0
                else:
                    self.top_value.value = top + length
            else:
                self.images[top: self.capacity] = images[:back_capacity]
                self.actions[top: self.capacity] = actions[:back_capacity]
                self.rewards[top: self.capacity] = rewards[:back_capacity]
                self.terminals[top: self.capacity] = terminals[:back_capacity]
                rest = length - back_capacity
                self.images[:rest] = images[back_capacity:]
                self.actions[:rest] = actions[back_capacity:]
                self.rewards[:rest] = rewards[back_capacity:]
                self.terminals[:rest] = terminals[back_capacity:]
                self.top_value.value = rest

    def random_batch(self, batch_size):

        images = np.zeros((batch_size,
                           self.phi_length + 1,
                           self.channel,
                           self.height,
                           self.width),
                          dtype='uint8')

        with self.lock:
            begin = 0
            end = self.capacity
            if self.size_value.value < self.capacity:
                end = self.top_value.value
            end = end - self.phi_length

            indices = self.rng.uniform(begin, end, (batch_size,)).astype('int32')

            for i in range(batch_size):
                target_begin = indices[i]
                target_end = target_begin + self.phi_length + 1
                # get (phi_length + 1) images
                images[i] = self.images[target_begin: target_end]

            actions = np.take(self.actions, indices, axis=0)
            rewards = np.take(self.rewards, indices, axis=0)
            terminals = np.take(self.terminals, indices, axis=0)

            return images, actions, rewards, terminals


def test_add():
    height = 10
    width = 5
    channel = 2
    phi_length = 4
    discount = 0.99
    capacity = 32
    rng = np.random.RandomState(100)

    mp_ctx = mp.get_context('spawn')
    buffer_data = create_replay_buffer_data(height, width, channel, phi_length, capacity, mp_ctx)

    buffer = ReplayBuffer(height, width, channel, phi_length, capacity, buffer_data)

    def get_experience(rows, begin=100):
        images = np.arange(begin, begin + (rows * channel * height * width)).reshape((rows, channel, height, width))
        actions = np.arange(begin, begin + rows)
        rewards = np.arange(begin, begin + rows)
        return rows, images.tolist(), actions.tolist(), rewards.tolist()

    buffer.add_experience(*get_experience(10))
    buffer.add_experience(*get_experience(10))
    buffer.add_experience(*get_experience(8))
    buffer.add_experience(*get_experience(8))

    print('image:\n', buffer.images)
    print('actions:\n', buffer.actions)
    print('rewards:\n', buffer.rewards)
    print('terminals:\n', buffer.terminals)

    print('capacity=', buffer.capacity)
    print('size=', buffer.size_value.value)
    print('top=', buffer.top_value.value)


def test_get():
    height = 10
    width = 5
    channel = 2
    phi_length = 4
    discount = 0.99
    capacity = 300
    rng = np.random.RandomState()

    mp_ctx = mp.get_context('spawn')
    buffer_data = create_replay_buffer_data(height, width, channel, phi_length, capacity, mp_ctx)

    buffer = ReplayBuffer(height, width, channel, phi_length, capacity, buffer_data)

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
    print('size=', buffer.size_value.value)
    print('top=', buffer.top_value.value)

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
