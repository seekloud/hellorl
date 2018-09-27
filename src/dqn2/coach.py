# Author: Taoz
# Date  : 9/16/2018
# Time  : 10:57 PM
# FileName: coach.py


import multiprocessing as mp
import signal

import src.utils as utils
from src.dqn2.config import *
from src.dqn2.judge import start_judge, SharedScreen
from src.dqn2.network import save_model_to_file
from src.dqn2.player import start_player
from src.dqn2.q_learning import QLearning
from src.dqn2.replay_buffer import ReplayBuffer
from src.dqn2.replay_buffer import create_replay_buffer_data
from src.ztutils import CirceBuffer


def start_coach():
    # create coach, and start it.
    pid = os.getpid()
    ppid = os.getppid()
    print('++++++++++++++++++ Coach starting.... pid=[%s] ppid=[%s]' % (str(pid), str(ppid)))
    coach = Coach()
    coach.start()
    print('Coach finish.')


_killed = False


def term(sig_num, addtion):
    print('sig_num', sig_num)
    print('addtion', addtion)
    global _killed
    _killed = True
    print('Main process [Coach] stopping...')


class Coach(object):

    def __init__(self):

        signal.signal(signal.SIGTERM, term)

        self.ctx = utils.try_gpu(GPU_INDEX)

        if os.name == 'nt':
            mp_method = 'spawn'
        elif os.name == 'posix':
            mp_method = 'forkserver'
        else:
            mp_method = 'forkserver'

        print('multiprocessing context method: ', mp_method)

        self.mp_ctx = mp.get_start_method(mp_method)

        self.report_queue = self.mp_ctx.Queue()
        self.process_list = []

        replay_buffer_data = \
            create_replay_buffer_data(HEIGHT, WIDTH, CHANNEL, BUFFER_MAX, self.mp_ctx)

        self.replay_buffer = ReplayBuffer(HEIGHT,
                                          WIDTH,
                                          CHANNEL,
                                          PHI_LENGTH,
                                          BUFFER_MAX,
                                          replay_buffer_data)

        self.player_agent_map, self.player_screen_map = \
            self._init_player(PLAYER_NUM, replay_buffer_data)

        self._init_judge(PRE_TRAIN_MODEL_FILE)

        self.last_data_time = time.time() + 100.0

        self.shared_play_net_version = self.mp_ctx.Value('i', -1)

        self.shared_play_net_file = PLAY_NET_MODEL_FILE
        self.episode_count = 0
        self.step_count = 0
        self.train_count = 0

        self.q_learning = QLearning(self.ctx, PRE_TRAIN_MODEL_FILE)

        self.stat_range = 100
        self.step_circe = CirceBuffer(self.stat_range)
        self.score_circe = CirceBuffer(self.stat_range)
        self.reward_circe = CirceBuffer(self.stat_range)

    """ 
    .
    .
    .
    .
    .
    """

    def _init_player(self, num, replay_buffer_data):
        player_id_list = [i for i in range(num)]

        player_agent_map = dict()
        player_screen_map = dict()

        for player_id in player_id_list:
            image_share = ()
            shared_screen_data = \
                SharedScreen.create_shared_data(image_share, PHI_LENGTH, self.mp_ctx)
            player_agent, judge_agent = self.mp_ctx.Pipe()

            p = self.mp_ctx.Process(target=start_player,
                                    args=(player_id,
                                          judge_agent,
                                          replay_buffer_data,
                                          self.report_queue,
                                          shared_screen_data,
                                          RANDOM_EPISODE_PER_PLAYER))
            p.start()
            self.process_list.append(p)
            player_agent_map[player_id] = player_agent
            player_screen_map[player_id] = shared_screen_data

        return player_agent_map, player_screen_map

    def _init_judge(self, model_file):

        p = self.mp_ctx.Process(target=start_judge,
                                args=(model_file,
                                      self.player_agent_map,
                                      self.player_screen_map,
                                      self.shared_play_net_version))
        p.start()
        self.process_list.append(p)

    def start(self):
        last_report = 0

        while not _killed:
            t0 = time.time()
            self._read_report()
            t1 = time.time()
            self._train()
            t2 = time.time()

            if (self.train_count + 1) % PLAY_NET_UPDATE_INTERVAL == 0:
                self._update_play_net()

            if (self.train_count + 1) % POLICY_NET_SAVE_INTERVAL == 0:
                self._save_policy_net()

            t3 = time.time()

            if (self.episode_count - last_report) > 1:
                last_report = self.episode_count

                print(
                    '\nCoach: episode=%d train=%d t1=%.3f t2=%.3f t_step=%d avg_step=%.2f avg_score=%.2f avg_reward=%.4f\n' % (
                        self.episode_count,
                        self.train_count,
                        (t1 - t0),
                        (t2 - t1),
                        self.step_count,
                        self.step_circe.avg(),
                        self.score_circe.avg(),
                        self.reward_circe.avg()
                    ))

            if time.time() - self.last_data_time > 100.0:
                print('Coach no data timeout: %.3f' % (time.time() - self.last_data_time))
                break

        print('[ WARNING ] ----------------------- Coach stop')
        for p in self.process_list:
            p.terminate()
        print('----------------')

    def _read_report(self):
        if not self.report_queue.empty():
            report = self.report_queue.get()
            (ep_step, ep_score, ep_reward) = report
            self.step_circe.add(ep_step)
            self.score_circe.add(ep_score)
            self.reward_circe.add(ep_reward)

            self.step_count += ep_step
            self.episode_count += 1

    def _train(self):
        if self.step_count > 100:
            self.train_count += 1
            bs = BATCH_SIZE
            t0 = time.time()
            images, actions, rewards, terminals = self.replay_buffer.random_batch(bs)
            t1 = time.time()
            loss = self.q_learning.train_policy_net(bs, images, actions, rewards, terminals)
            t2 = time.time()
            print('\nTrain time analysis: t1=%.4f t2=%.4f\n' % (t1 - t0, t2 - t1))
            return loss

    def _update_play_net(self):
        file_path = self.shared_play_net_file
        save_model_to_file(self.q_learning.policy_net, file_path)
        self.shared_play_net_version.value = self.train_count

    def _save_policy_net(self):
        current_time = time.strftime("%Y%m%d_%H%M%S")
        net = self.q_learning.policy_net
        file_path = MODEL_PATH + '/' + FILE_PREFIX + "_" + GAME_NAME + "_" + BEGIN_TIME + "_" + current_time + '.model'
        t0 = time.time()
        save_model_to_file(net, file_path)
        t1 = time.time()
        print('%s Save model file[%s] time[%.3fs]' % (time.strftime("%Y-%m-%d %H:%M:%S"), file_path, (t1 - t0)))

        return file_path
