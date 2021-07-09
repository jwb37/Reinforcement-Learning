from Params import Params

from . import Actions
from .Board import Board
from .PygameRenderer import PygameRenderer

from ..ActionSet import ActionSet
from utils.pygame import greyscale_tensor, rgb_tensor

import pygame


class RLEnv:
    RewardInfo = {
        "GameOver": -20,
        "ScoreExponent": 1.3,
        "ScoreFactor": 1
    }

    def __init__(self):
        self.board = Board(self)

        pygame.init()

        self.image_dimensions = Params.image_dims
        self.model_renderer = PygameRenderer(self)
        self.model_renderer.resize(*self.image_dimensions)

        if Params.isTrue('greyscale'):
            self.state_dims = (1, *self.image_dimensions)
            self.surface_to_tensor = greyscale_tensor
        else:
            self.state_dims = (3, *self.image_dimensions)
            self.surface_to_tensor = rgb_tensor

        num_actions = 6
        self.actions = ActionSet( [a for a in range(num_actions)] )
        self.actions.default = 0


    def notify_game_over(self):
        self.game_complete = True
        self.reward += self.RewardInfo["GameOver"]

    def notify_score(self, blocks_cleared):
        self.reward += self.RewardInfo["ScoreFactor"] * (blocks_cleared ** self.RewardInfo["ScoreExponent"])

    def reset(self):
        self.game_complete = False
        self.frame_count = 0
        self.board.start()

    def read_state(self):
        surface = pygame.Surface( self.image_dimensions )
        self.model_renderer.render_board( self.board, surface )
        return self.surface_to_tensor( surface )

    def update(self, action):
        self.reward = 0

        self.board.handle_action(action)
        self.board.update()
        self.frame_count += 1

        if self.isTest:
            self.human_renderer.render_board( self.board, self.display )
            pygame.display.update()
            self.clock.tick(60)

        return (self.reward, self.game_complete)

    def prepare_training(self):
        self.isTest = False

    def prepare_testing(self):
        self.isTest = True

        output_dimensions = (300, 600)
        self.human_renderer = PygameRenderer(self)
        self.human_renderer.resize( *output_dimensions )

        self.display = pygame.display.set_mode( output_dimensions )
        pygame.display.set_caption( "Tetris Attack" )
        self.clock = pygame.time.Clock()
