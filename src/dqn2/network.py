# Author: Taoz
# Date  : 9/18/2018
# Time  : 8:38 AM
# FileName: network.py


import mxnet as mx
from mxnet import init, nd
from mxnet.gluon import nn
from src.dqn2.config import *


def get_net(
        action_num: int,
        ctx: mx.Context):

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
    net.initialize(init.Xavier(), ctx=ctx)
    input_sample = nd.random.uniform(0, 255, (1, PHI_LENGTH * CHANNEL, HEIGHT, WIDTH), ctx=ctx) / 255.0
    net(input_sample)
    return net










