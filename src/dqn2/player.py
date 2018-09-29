# Author: Taoz
# Date  : 9/16/2018
# Time  : 10:56 PM
# FileName: player.py

import numpy as np

from src.dqn2.config import *
from src.dqn2.game_env import GameEnv
from src.dqn2.judge import SharedScreen
from src.dqn2.replay_buffer import ReplayBuffer
from src.ztutils import CirceBuffer
import traceback


def start_player(play_id: int,
                 judge_agent,
                 replay_buffer_data,
                 report_queue,
                 shared_screen_data,
                 random_episode: int
                 ):
    # create player and start it.
    try:

        pid = os.getpid()
        ppid = os.getppid()
        print('++++++++++++   Player[%d] starting. pid=[%s] ppid=[%s] ' % (play_id, str(pid), str(ppid)))
        player = Player(play_id,
                        judge_agent,
                        replay_buffer_data,
                        report_queue,
                        shared_screen_data,
                        random_episode)
        player.start()
    except Exception as ex:
        print('error: [%s]' % str(ex))
        traceback.print_exc()


class Player(object):
    def __init__(self,
                 play_id,
                 judge_agent,
                 replay_buffer_data,
                 report_queue,
                 shared_screen_data,
                 random_episode=10
                 ):
        self.player_id = play_id
        self.rng = np.random.RandomState(RANDOM_SEED + (play_id * 1000))
        self.game = GameEnv(game=GAME_NAME,
                            obs_type=OBSERVATION_TYPE,
                            frame_skip=FRAME_SKIP)

        self.action_num = ACTION_NUM
        self.random_episode = random_episode
        self.report_queue = report_queue

        self.judge_agent = judge_agent

        self.replay_buffer = ReplayBuffer(HEIGHT,
                                          WIDTH,
                                          CHANNEL,
                                          PHI_LENGTH, BUFFER_MAX,
                                          replay_buffer_data)
        image_shape = (CHANNEL, HEIGHT, WIDTH)
        self.shared_screen = SharedScreen(image_shape, PHI_LENGTH, shared_screen_data)
        self.experience_recoder = ExperienceRecorder()

        # self.episode_score_window = CirceBuffer(20)
        self.episode_steps_window = CirceBuffer(20)
        self.episode_count = 0
        self.total_step = 0
        self.rng = RANDOM
        self.epsilon = EPSILON_START
        print('Player[%d] init done.' % self.player_id)

    def run_episode(self, random_operation=False):
        ep_reward = 0.0
        ep_score = 0
        ep_step = 0

        observation = self.game.reset()

        # do no operation steps.
        max_no_op_steps = 5
        no_op_steps = self.rng.randint(PHI_LENGTH, PHI_LENGTH + max_no_op_steps + 1)
        for _ in range(no_op_steps):
            image = self.process_img(observation)
            observation, reward, episode_done, lives, score = self.game.step(0)
            # set shared screen mem.
            self.shared_screen.add_image(image)
            self.experience_recoder.add_step(image, 0, reward)

        episode_done = False
        t0 = time.time()
        while not episode_done:

            image = self.process_img(observation)

            if random_operation:
                action, q_val = self._random_action()
            else:
                action, q_val = self._choose_action(image)

            observation, reward, episode_done, lives, score = self.game.step(action)
            self.experience_recoder.add_step(image, action, reward)

            # print('player step[%d] %d %f' % (step_count, action, reward))
            ep_reward += reward
            ep_score += score
            ep_step += 1

        t1 = time.time()

        self.episode_steps_window.add(ep_step)
        self.episode_count += 1

        ep_report = (ep_step, ep_score, ep_reward)
        self.report_queue.put((self.player_id, ep_report))

        # record experience
        if ep_step >= 0:  # FIXME just for test.
            # if step_count >= self.episode_steps_window.avg():
            experience = self.experience_recoder.pop_experience()
            self.replay_buffer.add_experience(*experience)
        else:
            self.experience_recoder.clean()

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
                print('player[%d] got exception:[%s]' % (self.player_id, str(e)))
                traceback.print_exc()
                break
        print('[ WARNING ] ---- !!!!!!!!!! Player[%d] stop.' % self.player_id)

    def _choose_action(self, image):
        self.total_step += 1
        self.epsilon = max(EPSILON_MIN, self.epsilon - EPSILON_RATE)
        if self.rng.rand() < self.epsilon:
            action = self.rng.randint(0, self.action_num)
            q_val = 0.0
        else:
            # set shared screen mem.
            self.shared_screen.add_image(image)
            # print('player [%d] set image' % self.player_id)
            self.judge_agent.send(self.total_step)
            # print('player [%d] waiting action...' % self.player_id)
            msg = self.judge_agent.recv()
            action, q_val = msg
            # print('player [%d] got action [%d]' % (self.player_id, action))
        return action, q_val

    def _random_action(self):
        action = self.rng.randint(0, self.action_num)
        return action, 0.0

    @staticmethod
    def process_img(img):
        img = img.transpose(2, 0, 1)
        return img


class ExperienceRecorder(object):
    def __init__(self):
        self.images = None
        self.actions = []
        self.rewards = []
        pass

    def add_step(self, image: np.array, action, reward):
        target_shape = (1, *image.shape)
        image = image.reshape(target_shape)
        if self.images is None:
            self.images = image.copy()
        else:
            self.images = np.concatenate((self.images, image))

        self.actions.append(action)
        self.rewards.append(reward)

    def pop_experience(self):
        length = len(self.actions)
        experience = length, self.images, self.actions, self.rewards
        self.images = None
        self.actions = []
        self.rewards = []
        return experience

    def clean(self):
        self.images = None
        self.actions = []
        self.rewards = []


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
