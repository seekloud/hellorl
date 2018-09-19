# Author: Taoz
# Date  : 9/16/2018
# Time  : 10:56 PM
# FileName: player.py

import multiprocessing as mp
import queue

import random
import time
import numpy as np
from src.dqn2.game_env import GameEnv
from src.dqn2.config import *


def start_player(play_id: int,
                 observation_queue: queue.Queue,
                 action_in,
                 experience_queue: queue.Queue
                 ):
    # create player and start it.
    print('create player [%d]' % play_id)
    player = Player(play_id, observation_queue, action_in, experience_queue)
    while True:
        player.run_episode()


class Player(object):
    def __init__(self,
                 play_id,
                 observation_queue: queue.Queue,
                 action_in: mp.Pipe,
                 experience_queue: queue.Queue):
        self.rng = np.random.RandomState(RANDOM_SEED + (play_id * 1000))
        self.game = GameEnv(game=GAME_NAME,
                            obs_type=OBSERVATION_TYPE,
                            frame_skip=FRAME_SKIP)
        self.player_id = play_id
        self.observation_queue = observation_queue
        self.action_in = action_in
        self.experience_queue = experience_queue

        # self.episode_score_window = CirceBuffer(20)
        self.episode_steps_window = CirceBuffer(20)

    def run_episode(self):
        episode_score = 0
        step_count = 0

        images = []
        actions = []
        rewards = []
        terminals = []

        self.game.reset()

        # do no operation steps.
        observation = None

        max_no_op_steps = 20
        for _ in range(self.rng.randint(PHI_LENGTH, PHI_LENGTH + max_no_op_steps + 1)):
            observation, _, _, _, _ = self.game.step(0)

        episode_done = False
        while not episode_done:
            phi = images[-PHI_LENGTH:]
            action, q_val = self._choose_action(phi)
            images.append(observation.tolist())
            observation, reward, episode_done, lives, score = self.game.step(action)
            actions.append(action)
            rewards.append(reward)
            terminals.append(episode_done)

            episode_score += score
            step_count += 1

        # send experience to coach
        if step_count >= self.episode_steps_window.avg():
            experience = (images, actions, rewards, terminals)
            self.experience_queue.put(experience)

        self.episode_steps_window.add(step_count)

        return

    def _choose_action(self, phi):
        self.observation_queue.put((self.player_id, phi))
        action, q_val = self.action_in.get()
        return action, q_val


class CirceBuffer(object):
    def __init__(self, capacity: int):
        assert capacity > 0
        self._capacity = capacity
        self._list = []
        self._begin = 0
        self._sum = 0.0

    def add(self, num: float):
        self._sum += num
        if self.size() < self._capacity:
            self._list.append(num)
        else:
            self._sum -= self._list[self._begin]
            self._list[self._begin] = num
            self._begin = (self._begin + 1) % self._capacity

    def avg(self):
        length = self.size()
        if length > 0:
            return self._sum / length
        else:
            return 0.0

    def size(self):
        return len(self._list)

    def clean(self):
        self._begin = 0
        self._sum = 0.0
        self._list = []


def test_circe_buffer():
    buffer = CirceBuffer(5)
    buffer.add(1)
    print(buffer.avg())
    buffer.add(2)
    print(buffer.avg())
    buffer.add(3)
    print(buffer.avg())
    buffer.add(4)
    print(buffer.avg())
    buffer.add(5)
    print(buffer.avg())
    buffer.add(1)
    print(buffer.avg())
    buffer.add(2)
    print(buffer.avg())
    buffer.add(2)
    print(buffer.avg())
    buffer.add(2)
    print(buffer.avg())
    buffer.add(2)
    print(buffer.avg())
    buffer.add(2)
    print(buffer.avg())
    buffer.add(2)
    print(buffer.avg())
    pass


if __name__ == '__main__':
    pass
