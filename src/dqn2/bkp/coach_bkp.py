# Author: Taoz
# Date  : 9/16/2018
# Time  : 10:57 PM
# FileName: coach.py


import queue
import threading

import src.utils as utils
from src.dqn2.network import *
from src.dqn2.q_learning import QLearning
from src.dqn2.replay_buffer import ReplayBuffer
import os
from src.ztutils import CirceBuffer


def start_coach(pre_trained_model_file,
                episode_in_list,
                play_net_file,
                play_net_version
                ):
    # create coach, and start it.
    pid = os.getpid()
    ppid = os.getppid()
    print('++++++++++++++++++ Coach starting.... pid=[%s] ppid=[%s]' % (str(pid), str(ppid)))
    coach = Coach(pre_trained_model_file, episode_in_list, play_net_file, play_net_version)
    coach.start()
    print('Coach finish.')


def listen_experience(episode_in,
                      merge_queue: queue.Queue):
    print('Coach listen_experience by: ', threading.current_thread().name)
    while True:
        episode_info = episode_in.recv()
        merge_queue.put(episode_info)


class Coach(object):

    def __init__(self,
                 model_file: str,
                 experience_in_list: list,
                 shared_play_net_file,
                 shared_play_net_version):
        self.ctx = utils.try_gpu(GPU_INDEX)
        self.replay_buffer = ReplayBuffer(HEIGHT, WIDTH, CHANNEL, PHI_LENGTH, DISCOUNT, RANDOM, BUFFER_MAX)

        self.last_data_time = time.time() + 100.0

        self.experience_in_list = experience_in_list
        self.shared_play_net_version = shared_play_net_version
        self.shared_play_net_file = shared_play_net_file
        self.episode_count = 0
        self.step_count = 0
        self.train_count = 0
        self.local_episode_queue = queue.Queue(200)

        self.q_learning = QLearning(self.ctx, model_file)

        self.stat_range = 100
        self.step_circe = CirceBuffer(self.stat_range)
        self.score_circe = CirceBuffer(self.stat_range)
        self.reward_circe = CirceBuffer(self.stat_range)

        # start listener.
        count = 0
        for exp_in in self.experience_in_list:
            t = threading.Thread(target=listen_experience,
                                 args=(exp_in, self.local_episode_queue),
                                 name='queue_loader_' + str(count),
                                 daemon=False)
            t.start()
            count += 1

    def start(self):
        last_report = 0

        while True:
            t0 = time.time()
            self._read_experience()
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
        print('[ WARNING ] ----------------------- !!!!!!!!!! Coach stop')

    def _read_experience(self):
        qu = self.local_episode_queue
        timeout = 1.0
        episode_info = None
        t0 = time.time()
        t1 = 0.0
        t2 = 0.0
        t3 = 0.0
        t4 = 0.0
        t5 = 0.0
        t6 = 0.0
        t7 = 0.0
        t8 = 0.0
        if not qu.empty():
            try:
                t1 = time.time()
                episode_info = qu.get(timeout=timeout)
                t2 = time.time()
            except TimeoutError:
                episode_info = None
                print('[ WARNING ] read experience queue timeout.')

        if episode_info is not None:
            t3 = time.time()
            self.last_data_time = time.time()
            (player_id, experience, report) = episode_info
            t4 = time.time()
            if experience is not None:
                (length, images, actions, rewards) = experience
                t5 = time.time()
                self.replay_buffer.add_experience(length, images, actions, rewards)

            t6 = time.time()
            (ep_step, ep_score, ep_reward) = report
            t7 = time.time()

            self.step_circe.add(ep_step)
            self.score_circe.add(ep_score)
            self.reward_circe.add(ep_reward)

            self.step_count += ep_step
            self.episode_count += 1
            t8 = time.time()

            print('--------  ' +
                  'Time analysis: t1=%.4f t2=%.4f t3=%.4f t4=%.4f t5=%.4f t6=%.4f t7=%.4f t8=%.4f \n' % (
                      t1 - t0,
                      t2 - t1,
                      t3 - t2,
                      t4 - t3,
                      t5 - t4,
                      t6 - t5,
                      t7 - t6,
                      t8 - t7
                  ))

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
