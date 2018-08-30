# Author: Taoz
# Date  : 8/27/2018
# Time  : 11:33 PM
# FileName: test1.py

import mxnet as mx
from mxnet import init, nd, gluon
from mxnet.gluon import nn


ctx = mx.cpu()
import numpy



class DQNOutput(mx.operator.CustomOp):
    def __init__(self):
        super(DQNOutput, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # TODO Backward using NDArray will cause some troubles see `https://github.com/dmlc/mxnet/issues/1720'
        x = out_data[0].asnumpy()
        action = in_data[1].asnumpy().astype(numpy.int)
        reward = in_data[2].asnumpy()
        dx = in_grad[0]
        ret = numpy.zeros(shape=dx.shape, dtype=numpy.float32)
        ret[numpy.arange(action.shape[0]), action] \
            = numpy.clip(x[numpy.arange(action.shape[0]), action] - reward, -1, 1)
        self.assign(dx, req[0], ret)


@mx.operator.register("DQNOutput")
class DQNOutputProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(DQNOutputProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'action', 'reward']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        action_shape = (in_shape[0][0],)
        reward_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, action_shape, reward_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return DQNOutput()



def copy_params(src_net, dst_net):
    ps_src = src_net.collect_params()
    ps_dst = dst_net.collect_params()
    prefix_length = len(src_net.prefix)
    for k, v in ps_src.items():
        k = k[prefix_length:]
        v_dst = ps_dst.get(k)
        v_dst.set_data(v.data())


def test_net_paras_copy():
    net1 = get_net(10)
    net2 = get_net(10)

    input = nd.arange(2880).reshape((3, 3, 20, 16))

    net1(input)
    net2(input)

    ps1 = net1.collect_params()
    ps2 = net2.collect_params()

    print(str(net1))
    print('----------------------------------')
    print(str(net2))
    print('++++++++++++++++++++++')

    print(ps1)
    print('----------------------------------')
    print(ps2)
    print('++++++++++++++++++++++')

    print(net1.prefix)
    print('----------------------------------')
    print(net2.prefix)
    print('++++++++++++++++++++++')

    prefix_length = len(net2.prefix)

    print(ps1.keys())
    print('----------------------------------')
    print(ps2.keys())
    print('++++++++++++++++++++++')

    copy_params(net1, net2)

    print('++++++++++++++++++++++')

    print(net1.collect_params().values())
    print('----------------------------------')
    print(net2.collect_params().values())
    print('++++++++++++++++++++++')

    #
    # print(net1.collect_params().items())
    # print('----------------------------------')
    # print(net2.collect_params().items())
    # print('++++++++++++++++++++++')

    # net2.collect_params().update(net1.collect_params())
    # net2[0].collect_params().update(net1[0].collect_params())

    # print(net1[0].collect_params())
    # print('----------------------------------')
    # print(net2[0].collect_params())
    # print('++++++++++++++++++++++')

    # net2[0].collect_params().update(net1[0].collect_params())

    print(nd.sum(net1[0].weight.data() - net2[0].weight.data()).asnumpy)
    print(nd.sum(net1[1].weight.data() - net2[1].weight.data()).asnumpy)
    print(nd.sum(net1[2].weight.data() - net2[2].weight.data()).asnumpy)
    print(nd.sum(net1[4].weight.data() - net2[4].weight.data()).asnumpy)
    print(nd.sum(net1[5].weight.data() - net2[5].weight.data()).asnumpy)
    print('----------------------------------')

    pass


def get_net(action_num):
    net = nn.Sequential()
    with net.name_scope():
        net.add(
            nn.Conv2D(channels=32, kernel_size=8, strides=4, padding=4, activation='relu'),
            nn.Conv2D(channels=64, kernel_size=4, strides=2, padding=2, activation='relu'),
            nn.Conv2D(channels=64, kernel_size=3, strides=1, padding=1, activation='relu'),
            nn.Flatten(),
            nn.Dense(512, activation="relu"),
            nn.Dense(action_num)
        )
        net.initialize(init.Xavier(), ctx=ctx)
        return net


from src.dqn.replay_buffer import ReplayBuffer
import numpy as np


def test_replay_buffer():
    rng = np.random.RandomState()
    buffer = ReplayBuffer(5, 6, 3, rng, 0.9, 30)
    img = np.arange(0, 90).reshape((5,6,3))
    buffer.add_sample(img, 0, 0, False )
    buffer.add_sample(img, 1, 10, False )
    buffer.add_sample(img, 2, 0, False )
    buffer.add_sample(img, 3, 0, False )
    buffer.add_sample(img, 4, 10, False )
    buffer.add_sample(img, 4, 0, False )
    buffer.add_sample(img, 4, 0, False )
    buffer.add_sample(img, 4, -1, False )
    buffer.add_sample(img, 4, 0, False )
    buffer.add_sample(img, 5, 10, False )
    buffer.add_sample(img, 6, -1, True )

    print(buffer.actions)
    print(buffer.rewards)
    print(buffer.R)

    pass


if __name__ == '__main__':
    test_replay_buffer()
