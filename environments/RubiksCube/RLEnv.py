import math
import torch
import random
import numpy as np

from Params import Params

from .Cube import Cube
from ..ActionSet import ActionSet

class RLEnv:
    def __init__(self):
        if Params.isTrue('cube_side_length'):
            side_length = Params.cube_side_length
        else:
            side_length = 3

        self.cube = Cube(side_length)
        self.actions = ActionSet([i for i, _ in enumerate(self.cube.actions)])
        if Params.isTrue('cube_one_hot'):
            self.state_dims = (self.cube.blocks.size * 6,)
        else:
            self.state_dims = (self.cube.blocks.size,)

        if Params.isTrue('cube_max_steps'):
            self.max_steps = Params.cube_max_steps
        else:
            self.max_steps = math.inf

    def read_state(self):
        state = self.cube.state
        if Params.isTrue('cube_one_hot'):
            state = np.eye(6)[state].flatten()
        return torch.Tensor(state).unsqueeze(0)

    def prepare_testing(self):
        pass

    def prepare_training(self):
        pass

    def reset(self):
        self.num_steps = 0
        self.cube.reset()
        self.cube.scramble(
            random.randint(Params.cube_min_difficulty, Params.cube_max_difficulty)
        )

    def update(self, action):
        self.cube.take_action_by_idx(action)
        if self.cube.check_solved():
            return (Params.RewardInfo['solved'], True)

        self.num_steps += 1
        if self.num_steps > self.max_steps:
            return (Params.RewardInfo['stuck'], True)

        return (Params.RewardInfo['move'], False)
