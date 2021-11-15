from typing import (
    Tuple,
)

import random
import numpy as np
import torch
import SumTree

class SumTree:
    Date_Ptr = 0

    def __init__(self, channels, capacity, device, full_sink: bool = True):
        self.capacity = capacity
        self.device = device
        self.SumTree = np.zeros(2 * capacity - 1)  #有capacity个叶子结点

        sink = lambda x: x.to(device) if full_sink else x
        self.__m_states = sink(torch.zeros(
            (capacity, channels, 84, 84), dtype=torch.uint8))
        self.__m_actions = sink(torch.zeros((capacity, 1), dtype=torch.long))
        self.__m_rewards = sink(torch.zeros((capacity, 1), dtype=torch.int8))
        self.__m_dones = sink(torch.zeros((capacity, 1), dtype=torch.bool))

    def add(self, priority, folded_state, action, reward, done):
        node_index = self.Date_Ptr + self.capacity - 1
        self.__m_states[self.Date_Ptr] = folded_state
        self.__m_actions[self.Date_Ptr] = action
        self.__m_rewards[self.Date_Ptr] = reward
        self.__m_dones[self.Date_Ptr] = done
        self.update(node_index, priority)

    #更新叶子结点中的优先级
    def update(self, index, priority):
        diff = priority - self.SumTree[index]
        self.SumTree[index] = priority
        while index != 0:
            index = (index - 1) // 2
            self.SumTree[index] += diff

    #获取叶子结点
    def get_leaf(self, value):
        parent = 0

        while True:
            left = 2 * parent + 1
            right = left + 1
            if left >= len(self.SumTree):
                index = parent
                break
            else:
                if value <= self.SumTree[left]:
                    parent = left
                else:
                    value -= self.SumTree[left]
                    parent = right

        ptr = index - self.capacity + 1
        return index, self.SumTree[index], \
            self.__m_states[ptr, :4].to(self.device).float(), \
            self.__m_states[ptr, 1:].to(self.device).float(), \
            self.__m_actions[ptr].to(self.device), \
            self.__m_rewards[ptr].to(self.device).float(), \
            self.__m_dones[ptr].to(self.device)

    @property
    def total_p(self):
        return self.SumTree[0]