# Author: Taoz
# Date  : 9/26/2018
# Time  : 10:29 PM
# FileName: judger.py


import queue
import threading
from src.dqn2.network import get_net
from src.dqn2.config import *
import src.utils as utils
from src.dqn2.shared_utils import to_np_array, create_shared_data
from mxnet import nd
import signal
import numpy as np
import traceback


def start_judge(model_file,
                play_net_file_path,
                player_agent_map,
                player_screen_map,
                shared_play_net_version
                ):
    pid = os.getpid()
    ppid = os.getppid()
    print('++++++++++++   Judge starting... pid=[%s] ppid=[%s] ' % (str(pid), str(ppid)))
    judge = Judge(model_file,
                  play_net_file_path,
                  player_agent_map,
                  player_screen_map,
                  shared_play_net_version
                  )
    judge.start()


def listen_player(player_id,
                  player_agent,
                  shared_screen_data,
                  merge_queue: queue.Queue):
    print('Experiment listen_player by: ', threading.current_thread().name)
    image_shape = (CHANNEL, HEIGHT, WIDTH)
    shared_screen = SharedScreen(image_shape, PHI_LENGTH, shared_screen_data)
    while True:
        # FIXME num should be ignore?
        num = player_agent.recv()
        # print('judge got player[%d] msg: %s' % (player_id, num))
        observation = shared_screen.get_phi()
        merge_queue.put((player_id, observation))


_killed = False


def term(sig_num, addtion):
    print('sig_num', sig_num)
    print('addtion', addtion)
    global _killed
    _killed = True
    print('Judge stopping...')


class Judge(object):

    def __init__(self,
                 model_file,
                 play_net_file_path,
                 player_agents: dict,
                 shared_screen_data_map: dict,
                 shared_play_net_version):

        signal.signal(signal.SIGTERM, term)

        self.ctx = utils.try_gpu(GPU_INDEX)

        self.play_net = get_net(ACTION_NUM, self.ctx)

        if model_file is not None:
            print('%s: Judge read trained model from [%s]' % (time.strftime("%Y-%m-%d %H:%M:%S"), model_file))
            self.play_net.load_parameters(model_file, ctx=self.ctx)

        self.play_net_version = -1
        self.shared_play_net_version = shared_play_net_version
        self.play_net_file = play_net_file_path
        self.local_observation_queue = queue.Queue()
        self.player_agents = player_agents
        self.step_count = 0

        # listen to players.
        for player_id, player_agent in player_agents.items():
            shared_screen_data = shared_screen_data_map[player_id]
            self.player_agents[player_id] = player_agent
            t = threading.Thread(target=listen_player,
                                 args=(player_id,
                                       player_agent,
                                       shared_screen_data,
                                       self.local_observation_queue),
                                 name='player_listener_' + str(player_id),
                                 daemon=False)
            t.start()

    def start(self):
        last_report = 0

        try:
            while not _killed:
                player_list, observation_list = self._read_observations()
                obs_len = len(observation_list)
                if obs_len > 0:
                    #print('Judge obs_len:', obs_len)

                    # print('Exp observation_list: ', len(observation_list))
                    # t0 = time.time()
                    # print('choose_batch_action for %d players:%s' % (obs_len, player_list))
                    action_list, max_q_list = self.choose_batch_action(observation_list)
                    # t1 = time.time()
                    for p, action, q_value in zip(player_list, action_list, max_q_list):
                        # print('Exp send action[%d] to player[%d]' % (action, p))
                        out_pipe = self.player_agents[p]
                        out_pipe.send((action, q_value))
                    self.step_count += len(player_list)
                    # t2 = time.time()
                    # print('experiment get choose_batch_action for [%d] players, choose time=%.2f, send time=%.2f' %
                    #       (len(player_list), (t1 - t0), (t2 - t1)))
                    # print('-----------------------------------')
                if self.step_count - last_report > 10000:
                    print('Judge process steps:', self.step_count)
                    last_report = self.step_count
                self.update_play_net()
        except Exception as ex:
            print('Judge got exception')
            traceback.print_exc()

        print('[ WARNING ] ---- !!!!!!!!!!  Judge stopped.')

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
        actions = max_index.astype('int')
        # print('choose_batch_action actions.shape', actions.shape)

        max_q_list = nd.pick(out, actions, 1).asnumpy().tolist()
        return actions.asnumpy().tolist(), max_q_list

    def update_play_net(self):
        latest_version = self.shared_play_net_version.value
        if latest_version > self.play_net_version:
            # print('Judge update_play_net')
            self.play_net.load_parameters(self.play_net_file, ctx=self.ctx)

            self.play_net_version = latest_version
        return


class SharedScreen(object):
    def __init__(self, image_shape, phi_length, shared_data):
        shape = (phi_length, *image_shape)
        images_data, count_value, index_value = shared_data
        self.buffer = to_np_array(images_data, shape, 'uint8')
        self._count = count_value
        self._index = index_value
        self.phi_length = phi_length
        pass

    def get_phi(self):
        assert self._count.value >= self.phi_length, 'count=%d, phi_length=%d' % (self._count.value, self.phi_length)
        i = self._index.value
        if i == 0:
            phi = self.buffer
        else:
            p1 = self.buffer[:i]
            p2 = self.buffer[i:]
            phi = np.concatenate((p2, p1))
        return phi

    def add_image(self, image: np.ndarray):
        i = self._index.value
        self.buffer[i] = image
        self._index.value = (i + 1) % self.phi_length
        self._count.value += 1

    @staticmethod
    def create_shared_data(image_shape, phi_length, mp_ctx):
        shape = (phi_length, *image_shape)
        images_data = create_shared_data(mp_ctx, shape, 'uint8')
        count_value = mp_ctx.Value('i', 0)
        index_value = mp_ctx.Value('i', 0)
        return images_data, count_value, index_value
