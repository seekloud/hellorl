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


def save_model(net, name, prefix=FILE_PREFIX, postfix='model'):
    time_mark = time.strftime("%Y%m%d_%H%M%S")
    file_path = MODEL_PATH + '/' + prefix + '_' + name + '_' + time_mark + '.' + postfix
    net.save_parameters(file_path)
    print(time.strftime("%Y-%m-%d %H:%M:%S"), ' save model success:', file_path)
    return file_path


def save_model_to_file(net, file_path):
    net.save_parameters(file_path)
    print(time.strftime("%Y-%m-%d %H:%M:%S"), ' save model success:', file_path)
    return file_path


def copy_parameters(src_net, dst_net):
    ps_src = src_net.collect_params()
    ps_dst = dst_net.collect_params()
    prefix_length = len(src_net.prefix)
    for k, v in ps_src.items():
        k = k[prefix_length:]
        v_dst = ps_dst.get(k)
        v_dst.set_data(v.data())
