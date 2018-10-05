# Author: Taoz
# Date  : 9/16/2018
# Time  : 10:57 PM
# FileName: coach.py


import multiprocessing
import signal

import src.utils as utils
from src.dqn2.config import *
from src.dqn2.config import _print_conf
from src.dqn2.judge import start_judge, SharedScreen
from src.dqn2.network import save_model_to_file
from src.dqn2.player import *
from src.dqn2.q_learning import QLearning
from src.dqn2.replay_buffer import ReplayBuffer, create_replay_buffer_data
from src.ztutils import CirceBuffer
import time
import src.ztutils as ztutils


def start_coach():
    # create coach, and start it.
    pid = os.getpid()
    ppid = os.getppid()
    print('++++++++++++++++++ Coach starting.... pid=[%s] ppid=[%s]' % (str(pid), str(ppid)))
    coach = Coach()
    print('start coach waiting...')
    time.sleep(1)
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

        print('0 - ' * 20)

        _print_conf()

        signal.signal(signal.SIGTERM, term)
        print('0.1 - ' * 20)

        self.ctx = utils.try_gpu(GPU_INDEX)
        print('0.2 - ' * 20)

        if os.name == 'nt':
            mp_method = 'spawn'
        elif os.name == 'posix':
            mp_method = 'forkserver'
        else:
            mp_method = 'forkserver'

        print('multiprocessing context method: ', mp_method)

        print('1 - ' * 20)

        self.mp_ctx = multiprocessing.get_context(mp_method)
        print('2 - ' * 20)

        self.report_queue = self.mp_ctx.Queue()
        self.process_list = []
        self.shared_data_storage = []

        print('3 - ' * 20)
        self.shared_play_net_version = self.mp_ctx.Value('i', -1)

        self.replay_buffer_data = \
            create_replay_buffer_data(HEIGHT, WIDTH, CHANNEL, BUFFER_MAX, self.mp_ctx)

        # print('!!!!!!!!!!  type replay_buffer_data:', type(replay_buffer_data))

        print('4 - ' * 20)
        self.replay_buffer = ReplayBuffer(HEIGHT,
                                          WIDTH,
                                          CHANNEL,
                                          PHI_LENGTH,
                                          BUFFER_MAX,
                                          self.replay_buffer_data)

        print('5 - ' * 20)
        self.last_data_time = time.time() + 100.0

        print('6 - ' * 20)
        self.shared_play_net_file = PLAY_NET_MODEL_FILE
        self.episode_count = 0
        self.step_count = 0
        self.train_count = 0

        self.q_learning = QLearning(self.ctx, PRE_TRAIN_MODEL_FILE)

        self.stat_range = 100
        self.record_ep = 0
        self.time_circe = CirceBuffer(self.stat_range)
        self.step_circe = CirceBuffer(self.stat_range)
        self.score_circe = CirceBuffer(self.stat_range)
        self.reward_circe = CirceBuffer(self.stat_range)

        ztutils.mkdir_if_not_exist(MODEL_PATH)

        print('7 - ' * 20)
        self.player_agent_map, self.player_screen_map = \
            self._init_player(PLAYER_NUM, self.replay_buffer_data)

        print('8 - ' * 20)
        self._init_judge(PRE_TRAIN_MODEL_FILE)

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
            image_shape = (CHANNEL, HEIGHT, WIDTH)
            shared_screen_data = \
                SharedScreen.create_shared_data(image_shape, PHI_LENGTH, self.mp_ctx)
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
                                      self.shared_play_net_file,
                                      self.player_agent_map,
                                      self.player_screen_map,
                                      self.shared_play_net_version))
        p.start()
        self.process_list.append(p)

    def start(self):
        last_report = 0

        try:
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

                if (self.episode_count - last_report) > 100:
                    last_report = self.episode_count

                    print(
                        '\n[%s] Coach stat: episode=%d train=%d record_ep=%d t_step=%d avg_time=%.2f avg_step=%.2f avg_score=%.2f avg_reward=%.3f' % (
                            time.strftime("%Y-%m-%d %H:%M:%S"),
                            self.episode_count,
                            self.train_count,
                            self.record_ep,
                            self.step_count,
                            self.time_circe.avg(),
                            self.step_circe.avg(),
                            self.score_circe.avg(),
                            self.reward_circe.avg()
                        ))
                if time.time() - self.last_data_time > 100.0:
                    print('Coach no data timeout: %.3f' % (time.time() - self.last_data_time))
                    break
        except Exception as ex:
            print('Coach got exception:', ex)
            traceback.print_exc()
        print('[ WARNING ] ----------------------- Coach stop')
        for p in self.process_list:
            p.terminate()
        print('----------------')

    def _read_report(self):
        if not self.report_queue.empty():
            player_id, report = self.report_queue.get()
            (ep_count, ep_time, ep_record, ep_step, ep_score, ep_reward) = report
            # (begin, end, mark, ep_count, ep_step, ep_score, ep_reward) = report
            print(
                '\n[%s] Coach got report from player[%d]: ep_count=%d time=%.3f record=%s steps=%d score=%.2f reward=%.4f' %
                (time.strftime("%Y-%m-%d %H:%M:%S"), player_id, ep_count, ep_time, ep_record, ep_step, ep_score,
                 ep_reward))

            # FIXED for test.
            # self._check_buffer_only_for_debug(begin, end, player_id, ep_count)
            if ep_record:
                self.record_ep += 1

            self.time_circe.add(ep_time)
            self.step_circe.add(ep_step)
            self.score_circe.add(ep_score)
            self.reward_circe.add(ep_reward)
            self.last_data_time = time.time()
            self.step_count += ep_step
            self.episode_count += 1

    def _check_buffer_only_for_debug(self, begin, end, player_id, ep_count):
        mark = player_id * 100 + ep_count

        target_images, target_actions, target_rewards = self.replay_buffer.get_slice_only_for_debug(begin, end)

        expected_images = np.ones_like(target_images) * mark
        expected_actions = np.ones_like(target_actions) * mark
        expected_rewards = np.ones_like(target_rewards) * mark

        t1 = np.array_equal(target_images, expected_images)
        t2 = np.array_equal(target_actions, expected_actions)
        t3 = np.array_equal(target_rewards, expected_rewards)

        if not t1:
            print('\n!!!!  CHECK ERROR: player_id=%d ep_count=%d from %d to %d' % (player_id, ep_count, begin, end))
            print('expected_images=', expected_images)
            print('target_images=', target_images)

        if not t2:
            print('\n!!!!  CHECK ERROR: player_id=%d ep_count=%d from %d to %d' % (player_id, ep_count, begin, end))
            print('expected_actions=', expected_actions)
            print('target_actions=', target_actions)

        if not t3:
            print('\n!!!!  CHECK ERROR: player_id=%d ep_count=%d from %d to %d' % (player_id, ep_count, begin, end))
            print('expected_rewards=', expected_rewards)
            print('target_rewards=', target_rewards)
        rst = t1 and t2 and t3

        if rst:
            print('\n!!!!  CHECK SUCCESS: player_id=%d ep_count=%d from %d to %d' % (player_id, ep_count, begin, end))

        return rst

    def _train(self):
        if self.step_count > 100:
            self.train_count += 1
            bs = BATCH_SIZE
            t0 = time.time()
            images, actions, rewards, terminals = self.replay_buffer.random_batch(bs)
            t1 = time.time()
            loss = self.q_learning.train_policy_net(bs, images, actions, rewards, terminals)
            t2 = time.time()
            # print('\nTrain time analysis: t1=%.4f t2=%.4f step=%d\n' % (t1 - t0, t2 - t1, self.step_count))
            return loss
        else:
            print('waiting init steps...       ', self.step_count)
            time.sleep(2.0)

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
