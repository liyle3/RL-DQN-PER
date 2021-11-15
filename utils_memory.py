from typing import (
    Tuple,
)

import random
import numpy as np
import torch
import SumTree

from utils_types import (
    BatchAction,
    BatchDone,
    BatchNext,
    BatchReward,
    BatchState,
    TensorStack5,
    TorchDevice,
)

class ReplayMemory(object):
    epsilon = 0.01
    alpha = 0.4
    beta = 0.4
    increment_unit = 0.001
    max_error = 1.0

    def __init__(
            self,
            channels: int,
            capacity: int,
            device: TorchDevice,

    ) -> None:

        self.size = 0
        self.tree = SumTree(channels, capacity, device)


    def push(
            self,
            folded_state: TensorStack5,
            action: int,
            reward: int,
            done: bool,
    ) -> None:

        priority_max = np.max(self.tree.SumTree[-self.tree.capacity:])
        if priority_max == 0:
            priority_max = self.max_error
        self.tree.add(priority_max, folded_state, action, reward, done)

        self.tree.Date_Ptr += 1
        self.size = max(self.tree.Date_Ptr, self.size)
        self.tree.Date_Ptr %= self.tree.capacity

    def sample(self, batch_size: int):

        index = np.empty((batch_size,), dtype=np.int32)
        ISWeights = np.empty((batch_size, 1))

        b_state = []
        b_next = []
        b_action = []
        b_reward = []
        b_done = []

        interval_len = self.tree.total_p / batch_size   #区间长度
        self.beta = np.min([1., self.beta + self.increment_unit])

        probability_min = np.min(self.tree.SumTree[-self.tree.capacity:]) / self.tree.total_p
        #为保证分母不为零
        if probability_min == 0:
            probability_min = 0.00001

        for i in range(batch_size):
            a, b = interval_len * i, interval_len * (i + 1)

            while True:
                value = np.random.uniform(a, b)
                index, p, state, next, action, reward, done = self.tree.get_leaf(value)
                if p != 0.0:
                    break

            indices[i] = index
            probability = p / self.tree.total_p
            ISWeights[i, 0] = np.power(probability / probability_min, -self.beta)

            b_state.append(state)
            b_next.append(next)
            b_action.append(action)
            b_reward.append(reward)
            b_done.append(done)

        b_state = torch.stack(b_state)
        b_next = torch.stack(b_next)
        b_action = torch.stack(b_action)
        b_reward = torch.stack(b_reward)
        b_done = torch.stack(b_done)

        return indices, b_state, b_next, b_action, b_reward, b_done, batch, ISWeights

    def batch_update(self, index, error):
        error += self.epsilon
        clipped_error = np.minimum(error.cpu().data.numpy(), self.max_error)
        ps = np.power(clipped_error, self.alpha)
        for i, p in zip(index, ps):
            self.tree.update(i, p)

    def __len__(self) -> int:
        return self.size