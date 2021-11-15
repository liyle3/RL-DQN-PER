from typing import (
    Optional,
)

import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils_types import (
    TensorStack4,
    TorchDevice,
)

from utils_memory import ReplayMemory
from utils_model import DQN


class Agent(object):

    def __init__(
            self,
            action_dim: int,
            device: TorchDevice,
            gamma: float,
            seed: int,

            eps_start: float,
            eps_final: float,
            eps_decay: float,

            restore: Optional[str] = None,
    ) -> None:
        self.__action_dim = action_dim
        self.__device = device
        self.__gamma = gamma

        self.__eps_start = eps_start
        self.__eps_final = eps_final
        self.__eps_decay = eps_decay

        self.__eps = eps_start
        self.__r = random.Random()
        self.__r.seed(seed)

        self._policy = DQN(action_dim, device).to(device)
        self._target = DQN(action_dim, device).to(device)
        if restore is None:
            self._policy.apply(DQN.init_weights)
        else:
            self._policy.load_state_dict(torch.load(restore))
        self._target.load_state_dict(self._policy.state_dict())
        self.__optimizer = optim.Adam(
            self._policy.parameters(),
            lr=0.0000625,
            eps=1.5e-4,
        )
        self._target.eval()

    def run(self, state: TensorStack4, training: bool = False, testing: bool = False) -> int:
        """run suggests an action for the given state."""
        if training:
            self.__eps -= \
                (self.__eps_start - self.__eps_final) / self.__eps_decay
            self.__eps = max(self.__eps, self.__eps_final)

        if testing or self.__r.random() > self.__eps:
            with torch.no_grad():
                return self._policy(state).max(1).indices.item()
        return self.__r.randint(0, self.__action_dim - 1)

    def learn(self, memory: ReplayMemory, batch_size: int) -> float:
        """learn trains the value network via TD-learning."""
        indices, state_batch, next_batch, action_batch, reward_batch, done_batch, is_weights = memory.sample(batch_size)

        expected = []
        b_current_Q = self._policy(next_batch).cpu().data.numpy()
        next_action_op = np.argmax(b_current_Q, axis=1)
        b_target_Q = self._target(next_batch)

        for i in range(batch_size):
            if done_batch[i]:
                expected.append(reward_batch[i])
            else:
                target_Q_value = b_target_Q[i, next_action_op[i]]
                expected.append(reward_batch[i] + self.__gamma * target_Q_value)

        expected = torch.stack(expected)
        values = self._policy(state_batch).gather(1, action_batch)

        abs_error = torch.abs(expected - values)
        memory.batch_update(indices, abs_error)

        loss = (torch.FloatTensor(is_weights).to(self.__device) * F.mse_loss(values, expected)).mean()

        self.__optimizer.zero_grad()
        loss.backward()
        for param in self._policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.__optimizer.step()

        return loss.item()

    def sync(self) -> None:
        """sync synchronizes the weights from the policy network to the target
        network."""
        self._target.load_state_dict(self._policy.state_dict())

    def save(self, path: str) -> None:
        """save saves the state dict of the policy network."""
        torch.save(self._policy.state_dict(), path)
