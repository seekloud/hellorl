# Author: Taoz
# Date  : 9/16/2018
# Time  : 10:56 PM
# FileName: player.py

import numpy as np

from src.dqn2.config import *
from src.dqn2.game_env import GameEnv
from src.ztutils import CirceBuffer


def start_player(play_id: int,
                 action_chooser,
                 experience_out,
                 random_episode: int
                 ):
    # create player and start it.
    pid = os.getpid()
    ppid = os.getppid()
    print('++++++++++++   Player[%d] starting. pid=[%s] ppid=[%s] ' % (play_id, str(pid), str(ppid)))
    player = Player(play_id, action_chooser, experience_out, random_episode)
    player.start()


class Player(object):
    def __init__(self,
                 play_id,
                 action_chooser,
                 coach_agent,
                 random_episode=10
                 ):
        self.rng = np.random.RandomState(RANDOM_SEED + (play_id * 1000))
        self.game = GameEnv(game=GAME_NAME,
                            obs_type=OBSERVATION_TYPE,
                            frame_skip=FRAME_SKIP)
        self.action_num = ACTION_NUM
        self.player_id = play_id
        self.random_episode = random_episode

        self.action_chooser = action_chooser

        self.coach_agent = coach_agent

        # self.episode_score_window = CirceBuffer(20)
        self.episode_steps_window = CirceBuffer(20)
        self.episode_count = 0
        self.rng = RANDOM
        self.epsilon = EPSILON_START

    def run_episode(self, random_operation=False):
        ep_reward = 0.0
        ep_score = 0
        ep_step = 0

        images = []
        actions = []
        rewards = []

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
        t0 = time.time()
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
            ep_reward += reward
            ep_score += score
            ep_step += 1

        t1 = time.time()
        # send experience to coach
        experience = None
        if ep_step >= 0:  # FIXME just for test.
            # if step_count >= self.episode_steps_window.avg():
            experience = (len(images), images, actions, rewards)

        ep_report = (ep_step, ep_score, ep_reward)
        self.coach_agent.send((self.player_id, experience, ep_report))

        self.episode_steps_window.add(ep_step)
        self.episode_count += 1

        print('Player[%d] Episode[%d] done: time=%.3f step=%d score=%d reward=%.3f' %
              (self.player_id,
               self.episode_count,
               (t1 - t0),
               ep_step,
               ep_score,
               ep_reward
               ))

        return

    def start(self):
        count = 0
        while True:
            if count < self.random_episode:
                random_operation = True
            else:
                random_operation = False
            try:
                self.run_episode(random_operation=random_operation)

                count += 1
            except Exception as e:
                print('player[%d] got exception:%s, %s' % (self.player_id, str(e), str(e.__cause__)))
                break
        print('[ WARNING ] ---- !!!!!!!!!! Player[%d] stop.' % self.player_id)

    def _choose_action(self, phi):

        self.epsilon = max(EPSILON_MIN, self.epsilon - EPSILON_RATE)

        if self.rng.rand() < self.epsilon:
            action = self.rng.randint(0, self.action_num)
            q_val = 0.0
        else:
            # print('player [%d] send phi' % self.player_id)
            self.action_chooser.send(phi)
            # print('player [%d] waiting action...' % self.player_id)
            msg = self.action_chooser.recv()
            # print('msg:', msg)
            action, q_val = msg
            # print('player [%d] got action [%d]' % (self.player_id, action))

        return action, q_val

    def _random_action(self):
        action = self.rng.randint(0, self.action_num)
        return action, 0.0

    @staticmethod
    def process_img(img):
        img = img.transpose(2, 0, 1)
        return img.tolist()


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
