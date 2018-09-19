# Author: Taoz
# Date  : 9/16/2018
# Time  : 10:57 PM
# FileName: coach.py

import queue
import multiprocessing as mp
from src.dqn2.config import *
from src.dqn2.network import *
from src.dqn2.config import *
import src.utils as utils


def start_coach(pre_trained_model_file: str,
                experience_queue: queue.Queue,
                shared_inform
                ):
    # create coach, and start it.
    coach = Coach(pre_trained_model_file, experience_queue, shared_inform)
    print('Coach begin.')
    coach.start()
    print('Coach finish.')


class Coach(object):

    def __init__(self,
                 model_file: str,
                 experience_queue: queue.Queue,
                 shared_inform):
        self.ctx = utils.try_gpu(GPU_INDEX)
        self.policy_net = get_net(ACTION_NUM)
        self.target_net = get_net(ACTION_NUM)

        if model_file is not None:
            print('%s: read trained model from [%s]' % (time.strftime("%Y-%m-%d %H:%M:%S"), model_file))
            self.policy_net.load_parameters(model_file, ctx=self.ctx)

        copy_parameters(self.policy_net, self.target_net)

        self.experience_queue = experience_queue
        self.shared_inform = shared_inform
        self.episode_count = 0
        self.train_count = 0

    def start(self):
        while True:
            self._read_experience()
            self._train()

    def _read_experience(self):
        count = 0
        self.experience_queue.put(-1)
        experience = self.experience_queue.get()
        while not isinstance(experience, int):
            self._save_experience(experience)
            count += 1
            experience = self.experience_queue.get()
        while True:
            experience = self.experience_queue.get()
            if not isinstance(experience, int):
                self._save_experience(experience)
            elif experience == -1:
                break
            else:
                print('error experience_queue code:', experience)
                break
        print('Coach read episode =', count)

    def _save_experience(self, experiencc):
        # experience = (images, actions, rewards, terminals)
        self.episode_count += 1
        # TODO

        pass

    def _train(self):
        # TODO
        pass


def foo(a: int, qu: queue.Queue):
    print(a)
    print(qu)
    b: int = 5
    print(b)
    pass


if __name__ == '__main__':
    m = mp.Manager()
    q = m.Queue()
    foo(1, q)
