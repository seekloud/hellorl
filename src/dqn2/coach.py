# Author: Taoz
# Date  : 9/16/2018
# Time  : 10:57 PM
# FileName: coach.py


import src.utils as utils
from src.dqn2.network import *
from src.dqn2.q_learning import QLearning
from src.dqn2.replay_buffer import ReplayBuffer


def start_coach(pre_trained_model_file,
                experience_queue,
                play_net_version
                ):
    # create coach, and start it.
    print('++++++++++++++++++ Coach starting....')
    coach = Coach(pre_trained_model_file, experience_queue, play_net_version)
    coach.start()
    print('Coach finish.')


class Coach(object):

    def __init__(self,
                 model_file: str,
                 experience_in,
                 shared_play_net_version):
        self.ctx = utils.try_gpu(GPU_INDEX)
        self.replay_buffer = ReplayBuffer(HEIGHT, WIDTH, CHANNEL, PHI_LENGTH, DISCOUNT, RANDOM, BUFFER_MAX, self.ctx)

        self.experience_in = experience_in
        self.shared_play_net_version = shared_play_net_version
        self.episode_count = 0
        self.step_count = 0
        self.train_count = 0

        self.q_learning = QLearning(self.ctx, model_file)

    def start(self):
        while True:
            self._read_experience()
            self._train()
            if self.train_count + 1 % 2 == 0:
                self._update_play_net()

    # def _read_experience(self):
    #     count = 0
    #     while True:
    #         print(' - - Coach waiting...')
    #         experience = self.experience_queue.get()
    #         print(' - - Coach get data:')
    #         self._save_experience(experience)
    #         print(' - - Coach saved experience.')

    def _read_experience(self):
        count = 0
        while True:
            print(' - - Coach waiting...')
            t0 = time.time()
            experience = self.experience_in.recv()
            t1 = time.time()
            count += 1
            print(' - - Coach get data time:', (t1 - t0))
            self._save_experience(experience)
            print(' - - Coach saved experience. count =', count)

    # def _read_experience(self):
    #     count = 0
    #     self.experience_in.put(-1)
    #     while True:
    #         print('queue size:', self.experience_in.qsize())
    #         t0 = time.time()
    #         experience = self.experience_in.get()
    #         t1 = time.time()
    #         print(' - - - - - read queue time:', (t1 - t0))
    #         if not isinstance(experience, int):
    #             count += 1
    #             self._save_experience(experience)
    #         elif experience == -1:
    #             print('experience read done.')
    #             break
    #         else:
    #             print('error experience_queue code:', experience)
    #             break
    #     print(' - - - Coach read episode [%d], total step=[%d], total episode=[%d]' %
    #           (count, self.step_count, self.episode_count))
    #

    def _save_experience(self, experience):
        # experience = (step_count, images, actions, rewards)

        (player_id, length, images, actions, rewards) = experience
        print('coach got experience from player[%d], length=%d' % (player_id, length))
        # self.replay_buffer.add_experience(length, images, actions, rewards)
        self.step_count += length
        self.episode_count += 1

        print('save steps: %d' % length)

    def _train(self):
        # self.train_count += 1
        # bs = BATCH_SIZE
        # images, actions, rewards, terminals = self.replay_buffer.random_batch(bs)
        # loss = self.q_learning.train_policy_net(bs, images, actions, rewards, terminals)
        # return loss
        # time.sleep(1)
        print(' - Coach skip train in test...')

    def _update_play_net(self):

        # delete file first?
        # if os.path.exists(file_path):
        #     os.remove(file_path)

        # file_path = PLAY_NET_MODEL_FILE
        # save_model_to_file(self.q_learning.policy_net, file_path)
        # self.shared_play_net_version.value = self.train_count
        # time.sleep(0.1)
        print(' - Coach skip update_play_net..')

# def foo(a: int, qu: queue.Queue):
#     print(a)
#     print(qu)
#     b: int = 5
#     print(b)
#     pass
#
#
# if __name__ == '__main__':
#     m = mp.Manager()
#     q = m.Queue()
#     foo(1, q)
