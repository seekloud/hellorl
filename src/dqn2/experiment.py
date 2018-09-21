# Author: Taoz
# Date  : 9/16/2018
# Time  : 10:56 PM
# FileName: experiment.py


import multiprocessing as mp
import queue

import numpy as np
from mxnet import nd

import src.utils as utils
from src.dqn2.coach import start_coach
from src.dqn2.config import *
from src.dqn2.network import get_net
from src.dqn2.player import start_player


class Experiment(object):

    def __init__(self):

        self.model_file = PRE_TRAIN_MODEL_FILE
        self.ctx = utils.try_gpu(GPU_INDEX)

        self.play_net_version = -1
        self.play_net = get_net(ACTION_NUM, self.ctx)

        self.coach_play_net_version = mp.Value('i', -1)
        self.coach_play_net_file = PLAY_NET_MODEL_FILE

        if self.model_file is not None:
            print('%s: Experiment read trained model from [%s]' % (time.strftime("%Y-%m-%d %H:%M:%S"), self.model_file))
            self.play_net.load_parameters(self.model_file, ctx=self.ctx)

        self.step_count = 0
        return

    def start_train(self):
        print('+++++++++++++++++   start_train')
        worker_num = PLAYER_NUM

        pool = mp.Pool(worker_num + 1)
        manager = mp.Manager()

        player_observation_queue: queue.Queue = manager.Queue()

        experience_queue = manager.Queue()

        player_action_outs = dict()
        players = range(1, worker_num + 1)

        random_episode = RANDOM_EPISODE_PER_PLAYER
        if PRE_TRAIN_MODEL_FILE is not None:
            random_episode = 0

        # start players
        for player_id in players:
            print('set player:', player_id)
            action_in, action_out = mp.Pipe()
            pool.apply_async(start_player,
                             (player_id,
                              player_observation_queue,
                              action_in,
                              experience_queue,
                              random_episode)
                             )

            player_action_outs[player_id] = action_out
            print('player:', player_id, ' created.')

        # start coach
        pool.apply_async(start_coach, (self.model_file, experience_queue, self.coach_play_net_version))

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
                    print('Exp got req from player[%d]' % player_id)
                    observation_list.append(observation)
                    player_list.append(player_id)

            obs_len = len(observation_list)
            if obs_len > 0:
                print('observation_list: ', len(observation_list))

                action_list, max_q_list = self.choose_batch_action(observation_list)
                for p, action, q_value in zip(player_list, action_list, max_q_list):
                    print('Exp send action[%d] to player[%d]' % (action, p))
                    out_pipe = player_action_outs[p]
                    out_pipe.send((action, q_value))
                self.step_count += len(player_list)
                # print('experiment get observations count=', len(player_list))
                print('-----------------------------------')

            self.update_play_net()

    def update_play_net(self):
        latest_version = self.coach_play_net_version.value
        if latest_version > self.play_net_version:
            print('%s: Experiment updated play net from %d to %d]' % (
                time.strftime("%Y-%m-%d %H:%M:%S"), self.play_net_version, latest_version))
            self.play_net.load_parameters(self.coach_play_net_file, ctx=self.ctx)
            self.play_net_version = latest_version
        return

    # def choose_action(self, phi):
    #     shape0 = phi.shape
    #     state = nd.array(phi, ctx=self.ctx).reshape((1, -1, shape0[-2], shape0[-1]))
    #     out = self.play_net(state)
    #     max_index = nd.argmax(out, axis=1)
    #     action = max_index.astype(np.int).asscalar()
    #     # print('state:', state)
    #     # print('state s:', state.shape)
    #     # print('out:', out)
    #     # print('out s:', out.shape)
    #     # print('max_index:', max_index)
    #     # print('max_index s:', max_index.shape)
    #     # print('action:', action)
    #     # print('action type:', type(action))
    #
    #     max_q = out[0, action].asscalar()
    #     return action

    def choose_batch_action(self, phi_list):
        batch_input = nd.array(phi_list, ctx=self.ctx)

        print('choose_batch_action batch_input.shape', batch_input.shape)

        shape0 = batch_input.shape
        state = nd.array(batch_input, ctx=self.ctx).reshape((shape0[0], -1, shape0[-2], shape0[-1]))
        # print('choose_batch_action state.shape', state.shape)
        out = self.play_net(state)
        # print('choose_batch_action out.shape', out.shape)
        max_index = nd.argmax(out, axis=1)
        # print('choose_batch_action max_index.shape', max_index.shape)
        actions = max_index.astype(np.int)
        # print('choose_batch_action actions.shape', actions.shape)

        max_q_list = nd.pick(out, actions, 1).asnumpy().tolist()
        return actions.asnumpy().tolist(), max_q_list

    def start_test(self):
        pass


def train():
    exp = Experiment()
    exp.start_train()


if __name__ == '__main__':
    train()
    pass
