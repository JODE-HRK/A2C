# -*- coding:utf-8 -*-
# @Time : 2021/12/14 13:45
# @Athor : JODE
# @File : maze_env.py
# @Software: PyCharm
"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example. The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
import os

import Agent
import Entrance
import Item
import ItemExport
import numpy as np
import time
import sys

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

# 每一个单元格是40pixels的大小
UNIT = 20  # pixels
# 有20 * 20 个单元格
MAZE_H = 15  # grid height
MAZE_W = 21  # grid width


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['s', 'u', 'd', 'l', 'r']
        # self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 16
        self.title('maze')
        print(MAZE_H * UNIT)
        print(MAZE_W * UNIT)
        #                              纵向           横向
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))
        # self.geometry('800x600')
        self.agents = []
        self.agents_entity = []
        self.enters_entity = []
        self.ports_entity = []
        self.time_sec = 0
        self.carry_sum = 0
        self.dump_sum = 0
        self._build_maze()

    def add_time(self):
        self.time_sec += 1

    # 建立基本环境
    def _build_maze(self):
        # 设置背景为白色，框架大小为height和width
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)

        # 画线
        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

    def create_env(self):
        self._build_maze()

    # 为每一个实例创建位置
    def set_pos(self, posX, posY, color, kind):
        # print("Set Position")
        # create origin
        origin = np.array([UNIT / 2, UNIT / 2])
        item_center = origin + np.array([(posX - 1) * UNIT, (posY - 1) * UNIT])
        # item
        item = self.canvas.create_rectangle(
            item_center[0] - 5, item_center[1] - 5,
            item_center[0] + 5, item_center[1] + 5,
            fill=color
        )
        self.canvas.pack()
        # print(type(item))
        if kind == "agent":
            self.agents.append([item, posX, posY])
        return item

    # 建立模拟环境各种实例
    def prepare(self):

        self.time_sec = 0

        enters_pos = [[1, 5], [1, 11], [5, 1], [11, 1], [17, 1], [21, 5], [21, 11], [5, 15], [11, 15], [19, 15]]
        ports_pos = [[8, 4], [14, 4], [8, 11], [14, 11]]
        agents_pos = [[1, 1], [1, 2], [2, 1], [2, 2]]
        # agents_pos = [[1, 1]]

        for pos in enters_pos:
            num_id = self.set_pos(pos[0], pos[1], "green", "enter")
            e = Entrance.Entrance(num_id, pos[0], pos[1], "green", 5, 4)  # 设定入口为绿色，每5秒开始随机产生一个货物，最大上限为7个
            self.enters_entity.append(e)

        for pos in ports_pos:
            num_id = self.set_pos(pos[0], pos[1], "yellow", "port")
            p = ItemExport.Export(num_id, pos[0], pos[1], "yellow")
            self.ports_entity.append(p)

        for pos in agents_pos:
            num_id = self.set_pos(pos[0], pos[1], "red", "agent")
            a = Agent.Agent(num_id, pos[0], pos[1], "red")
            self.agents_entity.append(a)

    # 为智能体设置目标
    def set_aim(self):
        for a in self.agents_entity:
            if not a.aim_at:
                if not a.carry:
                    dis = 10000000
                    for entry in self.enters_entity:
                        if entry.get_item_num() - entry.aim_at_this > 0 and abs(entry.posX - a.posX) + abs(
                                entry.posY - a.posY) < dis:
                            dis = abs(entry.posX - a.posX) + abs(entry.posY - a.posY)
                            a.change_aim(entry)
                            entry.aim_at_this += 1
                            break
                            # x.aimX = entry.posX, x.aimY = entry.posY
                            # x.aim = entry
                elif a.carry is True:
                    dis = 10000000
                    for port in self.ports_entity:
                        if abs(port.posX - a.posX) + abs(port.posY - a.posY) < dis:
                            dis = abs(port.posX - a.posX) + abs(port.posY - a.posY)
                            a.change_aim(port)
                            break

    def generate_items(self):
        # 货物开始出现
        for e in self.enters_entity:
            if self.time_sec % e.Generation_cycle == 0:
                e.add()

    def reset(self):
        self.update()
        time.sleep(0.01)
        observation = []

        for p in self.ports_entity:
            self.canvas.delete(p.id)
        for e in self.enters_entity:
            self.canvas.delete(e.id)
        for a in self.agents_entity:
            self.canvas.delete(a.id)

        self.ports_entity.clear()
        self.agents_entity.clear()
        self.enters_entity.clear()
        self.dump_sum = 0
        self.carry_sum = 0

        # 画图和放置相关物品
        self.prepare()

        for _agent in self.agents_entity:
            if [_agent.aimX, _agent.aimY] is not [-1, -1]:
                observation.extend([_agent.posX, _agent.posY, _agent.aimX, _agent.aimY])
            else:
                observation.extend([_agent.posX, _agent.posY, MAZE_W, MAZE_H])

        return observation

    # 移动智能体
    def move(self, action, obj):

        s = self.canvas.coords(obj.id)

        move_done = False

        base_action = np.array([0, 0])
        # if self.action_space[action] is "s":
        #     move_done = True
        # elif self.action_space[action] is "u" and s[1] > UNIT:
        #     base_action[1] -= UNIT
        #     move_done = True
        #     obj.posY -= 1
        # elif self.action_space[action] is "d" and s[1] < (MAZE_H - 1) * UNIT:
        #     base_action[1] += UNIT
        #     move_done = True
        #     obj.posY += 1
        # elif self.action_space[action] is "l" and s[0] > UNIT:
        #     base_action[0] -= UNIT
        #     move_done = True
        #     obj.posX -= 1
        # elif self.action_space[action] is "r" and s[0] < (MAZE_W - 1) * UNIT:
        #     base_action[0] += UNIT
        #     move_done = True
        #     obj.posX += 1

        if self.action_space[action] is "s":
            move_done = True
        elif self.action_space[action] is "u":
            base_action[1] -= UNIT
            move_done = True
            obj.posY -= 1
        elif self.action_space[action] is "d":
            base_action[1] += UNIT
            move_done = True
            obj.posY += 1
        elif self.action_space[action] is "l":
            base_action[0] -= UNIT
            move_done = True
            obj.posX -= 1
        elif self.action_space[action] is "r":
            base_action[0] += UNIT
            move_done = True
            obj.posX += 1

        # Move agent
        x = self.canvas.move(obj.id, base_action[0], base_action[1])
        return move_done

    # 判断是不是该结束了
    def judge_env(self):
        if self.time_sec > 864000:
            return True
        for a in self.agents_entity:
            if a.posX > MAZE_W or a.posX <= 0 or a.posY > MAZE_H or a.posY <= 0:
                return True
            for a_ in self.agents_entity:
                if a == a_:
                    continue
                elif [a.posX, a.posY] == [a_.posX, a_.posY]:
                    return True
            for e in self.enters_entity:
                if [a.posX, a.posY] == [e.posX, e.posY]:
                    return True
            for p in self.ports_entity:
                if [a.posX, a.posY] == [p.posX, p.posY]:
                    return True

    # 计算reward
    def count_reward(self):
        add_reward = 0
        minus_reward = 0

        mo = [[0, 0, -1, 1], [1, -1, 0, 0]]

        for a in self.agents_entity:
            for i in range(4):
                if not a.carry and type(a.aim) == type(self.enters_entity[0]) and a.aimX == a.posX + mo[0][i] and a.aimY == a.posY + mo[1][i]:
                    a.carry_on()
                    item = a.aim.minus()
                    a.change_aim(self.ports_entity[item.get_aim_id()])
                    add_reward += 1
                    self.carry_sum += 1
                    break
                elif a.carry and type(a.aim) == type(self.ports_entity[0]) and a.aimX == a.posX + mo[0][i] and a.aimY == a.posY + mo[1][i]:
                    a.dump_off()
                    add_reward += 1
                    self.dump_sum += 1
                    break
        return add_reward

    def step(self, action, a_id):
        time.sleep(0.01)
        # self.time_sec += 1

        # enter生成物品
        self.generate_items()
        # Agent选择目的地
        self.set_aim()

        reward = 0

        done = self.move(action, self.agents_entity[a_id])

        observation = []

        coincidence_reward = self.count_reward()

        if (reward < 0 and coincidence_reward < 0) or (reward >= 0 and coincidence_reward >= 0):
            reward += coincidence_reward
        else:
            reward = min(reward, coincidence_reward)

        for a in self.agents_entity:
            # observation.extend([a.posX, a.posY, a.aimX, a.aimY]
            if [a.aimX, a.aimY] is not [-1, -1]:
                observation.extend([a.posX, a.posY, a.aimX, a.aimY])
            else:
                observation.extend([a.posX, a.posY, MAZE_W, MAZE_H])

        if reward < 0 or self.time_sec == 86400:
            done = True
        else:
            done = False
            # self.set_aim()
        return observation, reward, done

    def render(self):
        time.sleep(0.01)
        self.update()

    def judge_situtiation(self):
        for id, e in enumerate(self.enters_entity):
            print("第%d个进货口" % id)
            print("坐标(%d, %d)" % (e.posX, e.posY))
            print("%d个智能体的目标" % e.aim_at_this)
            print("生成%d个货物" % e.get_item_num())
            print("------------")

        print("————————————————")

        for id, a in enumerate(self.agents_entity):
            print("第%d个智能体" % id)
            print("当前坐标(%d, %d)" % (a.posX, a.posY))
            print("当前目标性质 %s" % type(a.aim))
            print("当前目标坐标(%d, %d)" % (a.aimX, a.aimY))
            print("当前是否携带货物：", a.carry)


# 环境调试main函数
if __name__ == '__main__':
    env = Maze()
    env._build_maze()
    # enter = Entrance.Entrance()
    env.prepare()
    t_action = [[2, 2, 2, 2],
                [2, 2, 2, 2],
                [4, 0, 4, 2],
                [2, 0, 2, 2],
                [2, 0, 2, 3],
                [2, 0, 3, 0],  # 此处以上为取货物操作  已检测全部都取到货物
                [4, 4, 4, 4],
                [1, 1, 1, 1],
                [4, 4, 4, 4],
                [4, 4, 4, 4],
                [4, 4, 4, 4],
                [4, 4, 4, 4],
                [4, 4, 0, 4],
                [4, 4, 0, 4]
                ]
    #
    # observation = env.reset()
    #
    # # actions = [1, 1, 1, 1]
    # actions = [1]
    #
    # print("NUM agents:")
    # print(len(env.agents_entity))
    #
    env.render()
    for x in t_action:
        observation_, reward, done = env.step(x)
        env.render()
        env.judge_situtiation()
        print("Action: %s" % x)
        print("Reward: %d" % reward)
        time.sleep(1)
        # os.system("pause")

    print(env.carry_sum)
    print(env.dump_sum)
    #
    # print(observation_, reward, done)
    # env.reset()

    # 设置位置（可知横轴为x，纵轴为y）
    # hell1 = env.set_pos(2, 3, "black", None)
    # hell2 = env.set_pos(3, 2, "green", None)
    # destination = env.set_pos(3, 3, "yellow", None)
    # agent = env.set_pos(4, 4, "red", None)
    #
    # print(agent)
    # agent, done = env.move(action="up", obj=agent)
    # print(agent), print(done)
    # run(agent)
    # print(agent.posX)
    # print(agent.posY)

    # env.after(10, run, agent)

    env.mainloop()
