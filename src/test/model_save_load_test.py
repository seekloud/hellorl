# Author: Taoz
# Date  : 9/15/2018
# Time  : 5:26 PM
# FileName: model_save_load_test.py


import os
import time
import mxnet as mx
from mxnet.gluon import nn
from mxnet import init, nd

PHI_LENGTH = 4
CHANNEL = 3
HEIGHT = 210
WIDTH = 160

ctx = mx.gpu(1)

INPUT_SAMPLE = nd.random.uniform(0, 255, (1, PHI_LENGTH * CHANNEL, HEIGHT, WIDTH), ctx=ctx) / 255.0


def get_net(action_num, input_sample):
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
    net(input_sample)
    return net


def load(file, policy_net):
    policy_net.load_parameters(file, ctx=ctx)
    print(time.strftime("%Y-%m-%d %H:%M:%S"), ' load model success:', file)
    pass


def save_params_to_file(policy_net, file):
    policy_net.save_parameters(file)
    print(time.strftime("%Y-%m-%d %H:%M:%S"), ' save model success:', file)


def save(file):
    pass


def test(file, times, out_dir):
    net = get_net(18, INPUT_SAMPLE)
    load(file, net)

    save_file = os.path.join(out_dir, 'tmp_model')

    print('load file:', file)
    print('save file:', save_file)
    save_params_to_file(net, save_file)

    t0 = time.time()
    for i in range(times):
        load(save_file, net)
    t1 = time.time()

    print('times=%d, time=%.3f' % (times, (t1 - t0)))


if __name__ == '__main__':
    target = '/home/zhangtao/model_file/hello_rl/save/net_dqn_20180904_162858_20180905_175454.model'
    t = 10
    out = 'tmp_test'
    test(target, t, out)

"""
测试结果
45M的net模型文件，
save，1.21s；
load，0.34s


load file: /home/zhangtao/model_file/hello_rl/save/net_dqn_20180904_162858_20180905_175454.model
save file: tmp_test/tmp_model
2018-09-15 17:59:38  save model success: tmp_test/tmp_model
2018-09-15 17:59:38  save model success: tmp_test/tmp_model
2018-09-15 17:59:38  save model success: tmp_test/tmp_model
2018-09-15 17:59:38  save model success: tmp_test/tmp_model
2018-09-15 17:59:38  save model success: tmp_test/tmp_model
2018-09-15 17:59:38  save model success: tmp_test/tmp_model
2018-09-15 17:59:39  save model success: tmp_test/tmp_model
2018-09-15 17:59:39  save model success: tmp_test/tmp_model
2018-09-15 17:59:39  save model success: tmp_test/tmp_model
2018-09-15 17:59:39  save model success: tmp_test/tmp_model
times=10, time=1.204
[zhangtao@g212 hellorl]$ ll tmp_test/
total 45464
-rw-rw-r--. 1 zhangtao zhangtao 46554390 Sep 15 17:59 tmp_model
[zhangtao@g212 hellorl]$ ll -h tmp_test/
total 45M
-rw-rw-r--. 1 zhangtao zhangtao 45M Sep 15 17:59 tmp_model

[zhangtao@g212 hellorl]$ python3 src/test/model_save_load_test.py 

2018-09-15 18:01:37  save model success: tmp_test/tmp_model
2018-09-15 18:01:37  load model success: tmp_test/tmp_model
2018-09-15 18:01:37  load model success: tmp_test/tmp_model
2018-09-15 18:01:37  load model success: tmp_test/tmp_model
2018-09-15 18:01:37  load model success: tmp_test/tmp_model
2018-09-15 18:01:37  load model success: tmp_test/tmp_model
2018-09-15 18:01:37  load model success: tmp_test/tmp_model
2018-09-15 18:01:37  load model success: tmp_test/tmp_model
2018-09-15 18:01:37  load model success: tmp_test/tmp_model
2018-09-15 18:01:37  load model success: tmp_test/tmp_model
2018-09-15 18:01:37  load model success: tmp_test/tmp_model
times=10, time=0.340

"""
