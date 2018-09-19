# Author: Taoz
# Date  : 9/16/2018
# Time  : 10:56 PM
# FileName: experiment.py


import multiprocessing as mp

import numpy as np
import mxnet as mx
from mxnet import nd

from src.dqn2.coach import start_coach
from src.dqn2.config import *
from src.dqn2.network import get_net
from src.dqn2.player import start_player
from src.dqn2.constants import *
import src.utils as utils


class Experiment(object):

    def __init__(self, model_file):
        self.ctx = utils.try_gpu(GPU_INDEX)
        self.play_net_file = model_file
        self.play_net = get_net(ACTION_NUM, self.ctx)

        if self.play_net_file is not None:
            print('%s: Experiment read trained model from [%s]' % (time.strftime("%Y-%m-%d %H:%M:%S"), model_file))
            self.play_net.load_parameters(model_file, ctx=self.ctx)

        self.step_count = 0
        return

    def start_train(self):
        worker_num = 5

        pool = mp.Pool(worker_num + 1)
        manager = mp.Manager()

        shared_inform_map = manager.dict()
        shared_inform_map[c_latest_model_file_key] = self.play_net_file

        player_observation_queue = manager.Queue()

        experience_queue = manager.Queue()

        player_action_outs = dict()
        players = range(1, worker_num + 1)

        # start players
        for player_id in players:
            action_in, action_out = mp.Pipe()
            pool.apply_async(start_player, (player_id, player_observation_queue, action_in))
            player_action_outs[player_id] = action_out

        # start coach
        pool.apply_async(start_coach, (self.play_net_file, experience_queue,  shared_inform_map))

        # process player observations
        while True:
            player_list = []
            observation_list = []
            player_observation_queue.put((-1, 0))
            while True:
                player_id, observation = player_observation_queue.get()
                if player_id == -1:
                    break
                else:
                    observation_list.append(observation)
                    player_list.append(player_id)

            action_list, max_q_list = self.choose_batch_action(observation_list)
            for p, action, q_value in zip(player_list, action_list, max_q_list):
                player_action_outs[p].send((action, q_value))

            self.step_count += len(player_list)

            self.update_play_net(shared_inform_map)

    def update_play_net(self, inform_map):
        latest_model_file = inform_map[c_latest_model_file_key]
        if latest_model_file != self.play_net_file:
            print('%s: Experiment updated play net from [%s]' % (time.strftime("%Y-%m-%d %H:%M:%S"), latest_model_file))
            self.play_net.load_parameters(latest_model_file, ctx=self.ctx)
            self.play_net_file = latest_model_file
        return

    def choose_action(self, phi):
        shape0 = phi.shape
        state = nd.array(phi, ctx=self.ctx).reshape((1, -1, shape0[-2], shape0[-1]))
        out = self.play_net(state)
        max_index = nd.argmax(out, axis=1)
        action = max_index.astype(np.int).asscalar()
        # print('state:', state)
        # print('state s:', state.shape)
        # print('out:', out)
        # print('out s:', out.shape)
        # print('max_index:', max_index)
        # print('max_index s:', max_index.shape)
        # print('action:', action)
        # print('action type:', type(action))

        max_q = out[0, action].asscalar()
        return action

    def choose_batch_action(self, phi_list):
        batch_input = nd.array(phi_list, ctx=self.ctx)
        shape0 = batch_input.shape
        state = nd.array(batch_input, ctx=self.ctx).reshape((1, -1, shape0[-2], shape0[-1]))
        out = self.play_net(state)
        max_index = nd.argmax(out, axis=1)
        actions = max_index.astype(np.int)
        # print('state:', state)
        # print('state s:', state.shape)
        # print('out:', out)
        # print('out s:', out.shape)
        # print('max_index:', max_index)
        # print('max_index s:', max_index.shape)
        # print('action:', action)
        # print('action type:', type(action))

        max_q_list = nd.pick(out, actions, 1).asnumpy().tolist()
        return actions.asnumpy().tolist(), max_q_list

    def start_test(self):
        pass



if __name__ == '__main__':
    exp = Experiment()
    exp.start_train()


    pass
