# Author: Taoz
# Date  : 9/16/2018
# Time  : 10:56 PM
# FileName: experiment.py


import multiprocessing
import queue

import numpy as np
from mxnet import nd

import src.utils as utils
from src.dqn2.coach import start_coach
from src.dqn2.config import *
from src.dqn2.config import _print_conf
from src.dqn2.network import get_net
from src.dqn2.player import start_player

import threading
import signal


def error_handle(name):
    def func(e: Exception):
        print('!!!!!!!!!!!    [%s] error happen: %s' % (name, str(e.__cause__)))

    return func


def listen_player(player_id,
                  player_agent,
                  merge_queue: queue.Queue):
    print('Experiment listen_player by: ', threading.current_thread().name)
    while True:
        observation = player_agent.recv()
        merge_queue.put((player_id, observation))


def term():
    print(' !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Experiment got kill -15')


class Experiment(object):

    def __init__(self):
        pid = os.getpid()
        ppid = os.getppid()
        print('++++++++++++++++++ Experiment starting.... pid=[%s] ppid=[%s]' % (str(pid), str(ppid)))

        self.model_file = PRE_TRAIN_MODEL_FILE
        self.ctx = utils.try_gpu(GPU_INDEX)

        self.play_net_version = -1
        self.play_net = get_net(ACTION_NUM, self.ctx)
        self.local_observation_queue = queue.Queue()

        self.coach_play_net_version = None
        self.coach_play_net_file = PLAY_NET_MODEL_FILE

        if self.model_file is not None:
            print('%s: Experiment read trained model from [%s]' % (time.strftime("%Y-%m-%d %H:%M:%S"), self.model_file))
            self.play_net.load_parameters(self.model_file, ctx=self.ctx)

        self.step_count = 0
        _print_conf()
        return

    def start_train(self):
        signal.signal(signal.SIGTERM, term)
        print('+++++++++++++++++   start_train')
        worker_num = PLAYER_NUM
        mp_ctx = multiprocessing.get_context('forkserver')
        pool = mp_ctx.Pool(worker_num + 1)

        manager = mp_ctx.Manager()

        self.coach_play_net_version = manager.Value('i', -1)

        # experience_queue = manager.Queue()

        player_action_outs = dict()
        experience_in_list = []
        players = range(1, worker_num + 1)

        random_episode = RANDOM_EPISODE_PER_PLAYER
        if PRE_TRAIN_MODEL_FILE is not None:
            random_episode = 0

        # start players
        for player_id in players:
            # print('set player:', player_id)
            player_agent, action_chooser = mp_ctx.Pipe()
            experience_in, experience_out = mp_ctx.Pipe()

            pool.apply_async(start_player,
                             (player_id,
                              action_chooser,
                              experience_out,
                              random_episode
                              ),
                             error_callback=error_handle("start_player")
                             )

            player_action_outs[player_id] = player_agent
            experience_in_list.append(experience_in)
            t = threading.Thread(target=listen_player,
                                 args=(player_id, player_agent, self.local_observation_queue),
                                 name='player_' + str(player_id),
                                 daemon=False)
            t.start()
            print('player:', player_id, ' created.')

        # start coach
        pool.apply_async(start_coach,
                         (self.model_file,
                          experience_in_list,
                          self.coach_play_net_file,
                          self.coach_play_net_version),
                         error_callback=error_handle("start_coach"))
        pool.close()

        # process player observations

        try:
            while True:
                player_list, observation_list = self._read_observations()
                obs_len = len(observation_list)
                if obs_len > 0:
                    # print('Exp observation_list: ', len(observation_list))
                    # t0 = time.time()
                    action_list, max_q_list = self.choose_batch_action(observation_list)
                    # t1 = time.time()
                    for p, action, q_value in zip(player_list, action_list, max_q_list):
                        # print('Exp send action[%d] to player[%d]' % (action, p))
                        out_pipe = player_action_outs[p]
                        out_pipe.send((action, q_value))
                    self.step_count += len(player_list)
                    # t2 = time.time()
                    # print('experiment get choose_batch_action for [%d] players, choose time=%.2f, send time=%.2f' %
                    #       (len(player_list), (t1 - t0), (t2 - t1)))
                    # print('-----------------------------------')

                self.update_play_net()
        except Exception as ex:
            print('experiment got exception: %s, %s' % (str(ex), str(ex.__cause__)))
        finally:
            pool.terminate()
            print('++++++++++++++++++++++++++++++++++++   Program exit. +++++++++++++++++++++\n\n\n')

    def _read_observations(self):
        player_list = []
        observation_list = []
        qu = self.local_observation_queue
        while not qu.empty():
            player_id, observation = self.local_observation_queue.get()
            # print('Exp got req from player[%d]' % player_id)
            observation_list.append(observation)
            player_list.append(player_id)
        return player_list, observation_list

    def update_play_net(self):
        latest_version = self.coach_play_net_version.value
        if latest_version > self.play_net_version:
            t0 = time.time()
            self.play_net.load_parameters(self.coach_play_net_file, ctx=self.ctx)
            t1 = time.time()
            print('%s: Experiment loaded play net from [%d] to [%d], time=%.3f]' % (
                time.strftime("%Y-%m-%d %H:%M:%S"),
                self.play_net_version,
                latest_version,
                t1 - t0))
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

        # print('choose_batch_action batch_input.shape', batch_input.shape)

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
