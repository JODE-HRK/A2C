# -*- coding:utf-8 -*-
# @Time : 2022/1/5 17:26
# @Athor : JODE
# @File : Item.py
# @Software: PyCharm

# 分拣系统中所有的实体的父类
class Item:
    def __init__(self, aimid):
        self.id = None
        self.aim_id = aimid

    def get_aim_id(self):
        return self.aim_id

