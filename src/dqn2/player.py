# Author: Taoz
# Date  : 9/16/2018
# Time  : 10:56 PM
# FileName: player.py

import multiprocessing as mp
import queue

import numpy as np

from src.dqn2.config import PHI_LENGTH, GAME_NAME, OBSERVATION_TYPE, FRAME_SKIP, RANDOM_SEED, ACTION_NUM
from src.dqn2.game_env import GameEnv


def start_player(play_id: int,
                 observation_queue: queue.Queue,
                 action_in,
                 experience_queue: queue.Queue,
                 random_episode: int
                 ):
    # create player and start it.
    print('++++++++++++   create player [%d]' % play_id)
    player = Player(play_id, observation_queue, action_in, experience_queue)
    count = 0
    while True:
        if count < random_episode:
            random_operation = True
        else:
            random_operation = False
        player.run_episode(random_operation=random_operation)
        count += 1


class Player(object):
    def __init__(self,
                 play_id,
                 observation_queue: queue.Queue,
                 action_in,
                 experience_queue: queue.Queue
                 ):
        self.rng = np.random.RandomState(RANDOM_SEED + (play_id * 1000))
        self.game = GameEnv(game=GAME_NAME,
                            obs_type=OBSERVATION_TYPE,
                            frame_skip=FRAME_SKIP)
        self.action_num = ACTION_NUM
        self.player_id = play_id
        self.observation_queue: queue.Queue = observation_queue
        self.action_in = action_in

        self.experience_queue: queue.Queue = experience_queue

        # self.episode_score_window = CirceBuffer(20)
        self.episode_steps_window = CirceBuffer(20)
        self.episode_count = 0

    def run_episode(self, random_operation=False):
        episode_score = 0
        step_count = 0

        images = []
        actions = []
        rewards = []
        # terminals = []

        observation = self.game.reset()

        # do no operation steps.
        max_no_op_steps = 5
        for _ in range(self.rng.randint(PHI_LENGTH, PHI_LENGTH + max_no_op_steps + 1)):
            obs = self.process_img(observation)
            images.append(obs)
            observation, reward, episode_done, lives, score = self.game.step(0)
            actions.append(0)
            rewards.append(reward)

        episode_done = False
        while not episode_done:
            phi = images[-PHI_LENGTH:]

            action, q_val = 0, 0.0

            if random_operation:
                action, q_val = self._random_action()
            else:
                action, q_val = self._choose_action(phi)

            obs = self.process_img(observation)
            images.append(obs)
            observation, reward, episode_done, lives, score = self.game.step(action)
            actions.append(action)
            rewards.append(reward)
            # print('player step[%d] %d %f' % (step_count, action, reward))
            # terminals.append(episode_done)

            episode_score += score
            step_count += 1

        # send experience to coach
        if step_count >= self.episode_steps_window.avg():
            experience = (step_count, images, actions, rewards)
            self.experience_queue.put(experience)

        self.episode_steps_window.add(step_count)

        self.episode_count += 1

        print('player[%d] finish episode[%d].' % (self.player_id, self.episode_count))

        return

    def _choose_action(self, phi):
        # print('player [%d] send phi' % self.player_id)
        self.observation_queue.put((self.player_id, phi))
        # print('player [%d] waiting action...' % self.player_id)
        msg = self.action_in.recv()
        # print('msg:', msg)
        action, q_val = msg
        # print('player [%d] get action [%d]' % (self.player_id, action))

        return action, q_val

    def _random_action(self):
        action = self.rng.randint(0, self.action_num)
        return action, 0.0

    @staticmethod
    def process_img(img):
        img = img.transpose(2, 0, 1)
        return img.tolist()


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
