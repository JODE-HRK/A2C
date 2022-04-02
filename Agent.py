# -*- coding:utf-8 -*-
# @Time : 2022/1/6 11:14
# @Athor : JODE
# @File : Agent.py
# @Software: PyCharm
from Item import Item
from Entrance import Entrance
from ItemExport import Export


class Agent(Item):
    def __init__(self, id, posX, posY, color):
        self.id = id
        self.posX = posX
        self.posY = posY
        self.color = color
        self.aim_at = False
        self.aimX = -1
        self.aimY = -1
        self.aim = None
        self.carry = False
        self.run_or_stop = "stop"

    def next_step(self, step):
        self.run_or_stop = step

    def running(self):
        return self.run_or_stop

    def is_carry(self):
        return self.carry

    def change_aim(self, aim):
        self.aim = aim
        self.aim_at = True
        self.aimX, self.aimY = aim.get_pos()

    def carry_on(self):
        self.aim_at = False
        self.carry = True
        # return self.item

    def dump_off(self):
        self.carry = False
        self.aim_at = False
        self.aim = None

    def toPos(self):
        return self.aimX, self.aimY
