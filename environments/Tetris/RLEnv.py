import torch
import pygame
import numpy as np


from Params import Params


from . import Actions
from . import BoardState
from .Game import Game
from .PygameRenderer import PygameRenderer


from ..ActionSet import ActionSet
from utils.pygame import rgb_tensor
from utils.pygame import greyscale_tensor


class RLEnv(Game):

    # Reward can be composed of the following
    # Penalise model for unnecessary movement:
    #   ActionPenalty & Penalised Actions
    # Penalise model for game over
    #   GameOver
    # Default scoring is 10, 20, 40, 80 for 1 line, 2 lines, 3 lines, tetris respectively
    #   ScoreScaling is a factor to multiply these values by
    # Reward model for landing pieces:
    #   PieceLanded
    # Penalise model for creating 'holes' (i.e. empty spaces covered by filled spaces)
    #   HoleCreated
    # Penalise model for increasing height of pieces placed on board
    #   MaxHeightChange (a function returning reward given 2 inputs: (old_height, new_height))
#    RewardInfo = {
#        'ActionPenalty': 0,
#        'PenalisedActions': { Actions.Left, Actions.Right, Actions.RotateLeft, Actions.RotateRight },
#        'GameOver': -30,
#        'ScoreScaling': 4,
#        'PieceLanded': 2,
#        'HoleCreated': -5,
#        'MaxHeightChange': (lambda old_height, new_height: 0 )      # Disable max height reward
#        'MaxHeightChange': (lambda old_height, new_height: -max(0,new_height-old_height)**1.5 )
#    }


    PermittedActions = (Actions.Nothing, Actions.Left, Actions.Right, Actions.RotateLeft, Actions.RotateRight)


    def __init__(self):
        super().__init__(height=10)

        pygame.init()

        self.image_dimensions = Params.image_dims

        self.model_renderer = PygameRenderer(self, bgcolour=(255,255,255))
        self.model_renderer.resize(*self.image_dimensions)

        self.actions = ActionSet(self.PermittedActions)
        self.actions.default = Actions.Nothing

        if Params.isTrue('greyscale'):
            self.state_dims = (1, *self.image_dimensions)
            self.surface_to_tensor = greyscale_tensor
        else:
            self.state_dims = (3, *self.image_dimensions)
            self.surface_to_tensor = rgb_tensor

        self.board.game = self
        self.difficulty = 29.5
        self.dynamic_difficulty = False

        self.RewardInfo = Params.RewardInfo

    def reset(self):
        super().start()
        self.reward = 0
        self.num_holes = 0
        self.max_height = 0

    #-------------------------------------------------------------------------
    # Reward Function Contributors

    def notify_score(self, points_scored):
        print( "Line completed" )
        self.reward += points_scored*self.RewardInfo['ScoreScaling']
        self.priority_update = True

    def notify_piece_landed(self):
        self.reward += self.RewardInfo['PieceLanded']

        holes_created = self.board.count_holes() - self.num_holes
        self.reward += holes_created * self.RewardInfo['HoleCreated']

        new_max_height = self.board.max_height()
        self.reward += self.RewardInfo['MaxHeightChange'](self.max_height, new_max_height)

    def notify_game_over(self):
        self.game_complete = True
        self.reward += self.RewardInfo['GameOver']

    #-------------------------------------------------------------------------
    def read_state(self):
        render_target = pygame.Surface( self.image_dimensions )
        self.model_renderer.render_board(render_target)
        return self.surface_to_tensor(render_target)

    def get_actions(self):
        return self.ActionSet

    def update(self, action):
        self.reward = 0
        self.max_height = self.board.max_height()
        self.num_holes = self.board.count_holes()

        if action in self.RewardInfo['PenalisedActions']:
            self.reward += self.RewardInfo['ActionPenalty']

        super().update( [action] )

        if self.isTest:
            self.test_renderer.render_board(self.display)
            pygame.display.update()

            # Lower than usual frame rate - will slow down disappearing line animations unnaturally,
            # but should allow a human to follow what the computer's doing in real time!
            self.clock.tick(20)
            print(action,self.reward)

        return (self.reward, self.game_complete)

    def prepare_training(self):
        self.isTest = False

    def prepare_testing(self):
        self.isTest = True

        display_dims = (300, 300)
        self.display = pygame.display.set_mode( display_dims )
        pygame.display.set_caption('Tetris')
        self.clock = pygame.time.Clock()

        self.test_renderer = PygameRenderer(self)
        self.test_renderer.resize( *display_dims )
