# With help from:
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import torch
from collections import namedtuple, deque
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
# import gym
from itertools import count
from dummyenv import EchoEnv

EXPERIENCE_REPLAY_SIZE = 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('prev_obs', 'action', 'obs', 'reward'))

# Memory for experience replay.
memory = deque(maxlen=EXPERIENCE_REPLAY_SIZE)


# The Deep Q-Learning function approximator, which is just a fully connected
# neural net with ReLU.
class FCN(nn.Module):
    # First (input) size should be the observation size.
    # Last (output) size should be the action space size.
    def __init__(self, sizes):
        super(FCN, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(curr, next)
                                     for curr, next in zip(sizes, sizes[1:])])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


env = EchoEnv()

BATCH_SIZE = 128
GAMMA = 0.0
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100000
TARGET_UPDATE = 100

# Get number of actions from gym action space
n_obs = env.observation_space.shape[0]  # posx, posy, velx, vely, angle, angleVel, leg1contact, leg2contact
n_actions = env.action_space.n  # ...

policy_net = FCN([n_obs, 2 * (n_obs + n_actions), n_actions]).to(device)
print(policy_net)
target_net = FCN([n_obs, 2 * (n_obs + n_actions), n_actions]).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr=0.01)


def epsilon_greedy(state, epsilon_greedy_annealing_step):
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * epsilon_greedy_annealing_step / EPS_DECAY)

    if random.random() > eps_threshold:
        with torch.no_grad():
            return torch.argmax(policy_net(torch.tensor(state))).item()
    else:
        return env.action_space.sample()
