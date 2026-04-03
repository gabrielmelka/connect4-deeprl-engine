# dqn_model.py
# network, buffer, action selection, training step
# fixes: deque buffer, legal action masking in target, double DQN, bigger network

import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


# network (bigger: 3 conv layers, 256 dense)
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 6 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )

    def forward(self, x):
        return self.net(x)


# replay buffer using deque (O(1) append and pop)
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, legal_next):
        # legal_next = list of legal actions in next_state (for masking in target)
        self.buffer.append((state, action, reward, next_state, done, legal_next))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, legal_nexts = zip(*batch)

        states      = torch.FloatTensor(np.array(states))
        actions     = torch.LongTensor(actions)
        rewards     = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones       = torch.FloatTensor(dones)

        return states, actions, rewards, next_states, dones, list(legal_nexts)

    def __len__(self):
        return len(self.buffer)


def choose_action(state, legal_actions, policy_net, epsilon):
    if random.random() < epsilon:
        return random.choice(legal_actions)
    with torch.no_grad():
        st = torch.FloatTensor(state).unsqueeze(0)
        qv = policy_net(st).squeeze(0)
        mask = torch.full((7,), float('-inf'))
        for a in legal_actions:
            mask[a] = 0.0
        qv = qv + mask
        return qv.argmax().item()


def train_step(policy_net, target_net, buffer, optimizer, gamma, batch_size):
    if len(buffer) < batch_size:
        return None

    states, actions, rewards, next_states, dones, legal_nexts = buffer.sample(batch_size)

    # Q(s, a) from policy net
    q_all  = policy_net(states)
    q_pred = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # double DQN: policy_net selects the action, target_net evaluates it
        q_policy_next = policy_net(next_states)   # for action selection
        q_target_next = target_net(next_states)   # for evaluation

        # mask illegal actions in next_state BEFORE taking argmax
        for i, legal in enumerate(legal_nexts):
            illegal_mask = torch.ones(7, dtype=torch.bool)
            for a in legal:
                illegal_mask[a] = False
            q_policy_next[i][illegal_mask] = float('-inf')

        # double DQN: select with policy, evaluate with target
        best_actions = q_policy_next.argmax(dim=1)
        q_next_max = q_target_next.gather(1, best_actions.unsqueeze(1)).squeeze(1)

        target = rewards + gamma * q_next_max * (1 - dones)

    loss = nn.SmoothL1Loss()(q_pred, target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()
