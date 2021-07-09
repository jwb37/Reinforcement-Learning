import torch
import random

from Params import Params
from collections import deque

class ExperienceBuffer:

    def __init__(self):
        self.store = deque([], Params.exp_buffer_size)

    def append(self, t):
        self.store.append(t)

    def sample(self, requested_size, device='cpu'):
        sample_size = min(len(self.store), requested_size)
        sample = random.sample(self.store, sample_size)

        old_states, new_states, actions, rewards, gameovers = zip(*sample)

        old_states = torch.cat(old_states).to(device)
        actions = torch.tensor(actions).to(device)
        new_states = torch.cat(new_states).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).T.to(device)
        gameovers = torch.tensor(gameovers, dtype=torch.bool).T.to(device)

        return (old_states, new_states, actions, rewards, gameovers)

    def save(self, path, print_message=False):
        torch.save(self.store, path)

        if print_message:
            print( "%d steps saved"%len(self.store) )

    def load(self, path):
        self.store = torch.load(path)
