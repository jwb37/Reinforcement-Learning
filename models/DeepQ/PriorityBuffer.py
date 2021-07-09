import torch
import random

from collections import deque

class PriorityBuffer:

    BufferLength = 20

    def __init__(self, size, device, discount_factor=0.997):
        self.generic_steps = deque([],size)
        self.priority_steps = deque([],size)
        self.buffer = deque([], self.BufferLength)
        self.discount_factor = discount_factor
        self.device = device

    def append(self, t):
        if len(self.buffer) == self.BufferLength:
            self.generic_steps.append(self.buffer[0])
        self.buffer.append(t)

    def append_priority(self, t):
# TODO - Potential idea: add list of known qvals to tuple
#        This will require some changes to the model

#        running_reward = final_reward = t[3]
#        for k in range(4,-1,-1):
#            running_reward *= self.discount_factor
#            self.buffer[k][5] = running_reward

        self.buffer.append(t)
        for data in self.buffer:
            self.priority_steps.append(data)
        self.buffer.clear()

    def sample(self, requested_size):
        num_priority = int(min(len(self.priority_steps),requested_size/2))
        num_generic = min(len(self.generic_steps), requested_size - num_priority)

        priority_sample = random.sample(self.priority_steps, num_priority)
        generic_sample = random.sample(self.generic_steps, num_generic)

        sample = priority_sample + generic_sample
        old_states, new_states, actions, rewards, gameovers = zip(*sample)

        old_states = torch.cat(old_states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        new_states = torch.cat(new_states).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).T.to(self.device)
        gameovers = torch.tensor(gameovers, dtype=torch.bool).T.to(self.device)

        return (old_states, new_states, actions, rewards, gameovers)

    def save(self, path, print_message=False):
        save_info = {
            'generic_steps': self.generic_steps,
            'priority_steps': self.priority_steps
        }

        torch.save(save_info, path)

        if print_message:
            print( "%d generic steps saved"%len(self.generic_steps) )
            print( "%d priority steps saved"%len(self.priority_steps) )


    def load(self, path):
        info = torch.load(path)

        self.generic_steps = info['generic_steps']
        self.priority_steps = info['priority_steps']
