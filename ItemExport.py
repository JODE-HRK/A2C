# -*- coding:utf-8 -*-
# @Time : 2022/1/4 13:44
# @Athor : JODE
# @File : ItemExport.py
# @Software: PyCharm
from Item import Item


class Export(Item):
    def __init__(self, id, posX, posY, color):
        # super(Export, self).__init__()
        self.id = id
        self.posX = posX
        self.posY = posY
        self.color = color
        self.sum = 0

    def get_pos(self):
        return self.posX, self.posY
