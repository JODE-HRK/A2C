# -*- coding:utf-8 -*-
# @Time : 2022/1/4 13:06
# @Athor : JODE
# @File : Entrance.py
# @Software: PyCharm
import random
from queue import Queue

from Item import Item

entrance_sum = 0


class Entrance(Item):
    def __init__(self, id, posX, posY, color, rate, max_num):
        # super(Entrance, self).__init__()
        self.id = id
        self.posX = posX
        self.posY = posY
        self.color = color
        self.Generation_cycle = rate  # 货物生成周期单位秒
        self.items = Queue(maxsize=max_num)
        self.aim_at_this = 0

    # 每隔几秒钟产生货物, 并设定最多能容纳的货物
    def add(self):
        # 50%的生成概率
        # print("YYDSYYDSYYDSYYDSYYDSYYDSYYDSYYDSYYDSYYDSYYDSYYDSYYDSYYDSYYDSYYDSYYDSYYDSYYDSYYDSYYDSYYDSYYDS")
        x = random.randint(0, 9)
        id = random.randint(0, 3)  # 闭区间

        # id = 0
        # if self.posX == 1 and self.posY == 5:
        #     item = Item(id)
        #     if not self.items.full():
        #         self.items.put(item)

        if x % 2 == 1 and not self.items.full():
            item = Item(id)
            self.items.put(item)

    # 货物被取走
    def minus(self):
        return self.items.get()

    def get_item_num(self):
        return self.items.qsize()

    def get_pos(self):
        return self.posX, self.posY
