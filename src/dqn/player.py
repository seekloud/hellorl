# Author: Taoz
# Date  : 8/25/2018
# Time  : 12:22 PM
# FileName: player.py

import numpy as np


class Player(object):
    def __init__(self, game, policy_net, rng, epsilon_min=0.10, epsilon_start=1.0, epsilon_decay=10000):
        self.game = game
        self.action_num = self.game.action_num()  # [0,1,2,..,action_num-1]
        self.policy_net = policy_net
        self.rng = rng
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_rate = (epsilon_start - epsilon_min) * 1.0 / epsilon_decay

    def run_episode(self, epoch, max_steps, replay_buffer, render=False, random_action=False):
        episode_step = 0
        episode_reword = 0
        st = self.game.reset()
        print('++++++ run_episode max_step=', max_steps)
        while True:
            # print('run step: %d, %d' % (epoch, episode_step))

            if random_action:
                action = self.game.random_action()
            else:
                action = self._choose_action(st)

            next_st, reward, episode_done, lives = self.game.step(action)
            terminal = episode_done or episode_step >= max_steps

            replay_buffer.add_sample(st, action, reward, terminal)

            episode_step += 1
            episode_reword += reward
            st = next_st
            if terminal:
                break
            if render:
                self.game.render()

        print('Episode end: %d, %d' % (epoch, episode_step))
        return episode_step, episode_reword

    def _choose_action(self, observation):
        self.epsilon = min(self.epsilon_min, self.epsilon - self.epsilon_rate)
        if self.rng.rand() < self.epsilon:
            action = self.rng.randint(0, self.action_num)
        else:
            action = self.policy_net.choose_action(observation)
        return action

    def update_policy_net(self, policy_net):
        self.policy_net = policy_net


if __name__ == '__main__':
    pass
