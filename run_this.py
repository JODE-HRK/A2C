# -*- coding:utf-8 -*-
# @Time : 2021/12/14 13:47
# @Athor : JODE
# @File : run_this.py
# @Software: PyCharm
import time

import numpy as np
import os
from maze_env import Maze
from RL_brain_selfwrite import Advantage_Actor_Critic

results = []

RENDER = False


def run_maze():
    # 将所有实例全部安放
    # env.prepare()

    for episode in range(100):
        # initial observation
        global RENDER
        RENDER = False
        s = env.reset()
        t = 0
        r_stack = []
        # print("Run_MAZE:")
        # print(observation)

        while True:
            # fresh env
            if RENDER:
                time.sleep(0.1)
                env.add_time()
                env.render()
                r_stack.clear()
                RENDER = False
                print(RENDER)

            # RL choose action based on observation
            # s.append(t % len(env.agents_entity))
            a = RL.choose_action(np.array(s))

            s_, r, done = env.step(a, t % len(env.agents_entity))

            # s_.append((t + 1) % len(env.agents_entity))

            done = False

            if t % len(env.agents_entity) == len(env.agents_entity) - 1:
                RENDER = True
                done = env.judge_env()
            if done:
                r = -20

            # 记录奖励
            r_stack.append(r)
            # 使用critic网络学习td error
            td_error = RL.critic_learn(np.array(s), r, np.array(s_))  # gradient = grad[r + gamma * V(s_) - V(s)]
            # 利用critic网络输出td error学习actor
            RL.actor_learn(np.array(s), a, td_error)  # true_gradient = grad[logPi(s,a) * td_error]
            # 状态变动
            s = s_
            t += 1
            if done:
                # 当前回合获得的总奖励
                ep_rs_sum = sum(r_stack)
                if 'running_reward' not in locals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01

                if running_reward > DISPLAY_REWARD_THRESHOLD:
                    RENDER = True  # rendering

                print("episode:", episode, "  reward:", int(running_reward))
                break

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RENDER = False
    DISPLAY_REWARD_THRESHOLD = 200
    RL = Advantage_Actor_Critic(n_features=env.n_features, n_actions=env.n_actions)
    env.after(1000, run_maze)
    env.mainloop()
    # for id, x in enumerate(results):
    #     print("The %d try is " % id)
    #     print(x)
    # RL.plot_cost()
