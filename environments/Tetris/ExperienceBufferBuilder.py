import torch
import pygame
import random
import numpy as np

from collections import deque

import Actions
import Rewards
import BoardState

from Game import Game
from PriorityBuffer import PriorityBuffer
from PygameRenderer import PygameRenderer


class ExperienceBufferBuilder(Game):

    # Only allow the model to observe state and take an action every n frames
    # Note that there will be many repeated frames in the game while animations occur etc.
    ActionFrameSkip = 1

    DiscountFactor = 0.99

    def __init__(self):
        super().__init__()

        pygame.init()

        self.image_dimensions = (50, 100)

        self.model_renderer = PygameRenderer(self, bgcolour=(255,255,255))
        self.model_renderer.resize(*self.image_dimensions)

        self.device = 'cuda'
        self.experience_buffer = PriorityBuffer(10000)

        self.board.game = self
        self.difficulty = 29
        self.dynamic_difficulty = False

    def start(self):
        super().start()

    #-------------------------------------------------------------------------
    # Reward Function Contributors

    def notify_score(self, points_scored):
        print( "Line cleared" )
        self.reward += Rewards.line_cleared(points_scored)
        self.priority_update = True

    def notify_piece_landed(self):
        self.reward += Rewards.piece_landed(self.board)

    #-------------------------------------------------------------------------

    def read_state(self):
        render_target = pygame.Surface( self.image_dimensions )

        self.model_renderer.render_board(render_target)

        pixels = pygame.surfarray.array3d(render_target).astype(np.float32)

        greyscale = pixels.dot([0.298, 0.587, 0.114])
        greyscale = greyscale / 255.0
        greyscale = torch.tensor(greyscale, dtype=torch.float).to(self.device)
        greyscale = greyscale.unsqueeze(0)
        greyscale = greyscale.unsqueeze(0)

        return greyscale

    def update(self, action):
        self.reward += Rewards.action_taken(action)

        super().update( [action] )

        if self.board.game_over:
            self.reward += Rewards.game_over()

        self.state = self.read_state()

    #-------------------------------------------------------------------------------------------

    def prepare_run(self):
        output_dimensions = (200, 400)

        self.display = pygame.display.set_mode( output_dimensions )
        pygame.display.set_caption('Tetris')
        self.clock = pygame.time.Clock()

        self.test_renderer = PygameRenderer(self)
        self.test_renderer.resize( *output_dimensions )


    def run(self, path):
        self.start()
        self.state = self.read_state()
        total_reward = 0

        while not self.game_complete:
            self.reward = 0
            before_state = self.state
            self.priority_update = False

            action = Actions.Nothing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.complete = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action = Actions.Left
                    elif event.key == pygame.K_RIGHT:
                        action = Actions.Right
#                    elif event.key == pygame.K_DOWN:
#                        action = Actions.FastFallOn
                    elif event.key == pygame.K_x:
                        action = Actions.RotateLeft
                    elif event.key == pygame.K_z:
                        action = Actions.RotateRight
                elif event.type == pygame.KEYUP:
                    pass
#                    if event.key == pygame.K_DOWN:
#                        action = Actions.FastFallOff

            if (self.frame_count % self.ActionFrameSkip) == 0 and self.board.state not in (BoardState.Clearing, BoardState.Falling):
                self.update(action)
                after_state = self.state
                action_index = torch.tensor( [self.ActionSet.index(action)] )

                if self.priority_update:
                    self.experience_buffer.append_priority (
                        (before_state, after_state, action_index, self.reward, self.game_complete)
                    )
                else:
                    self.experience_buffer.append (
                        (before_state, after_state, action_index, self.reward, self.game_complete)
                    )
            else:
                self.update(None)

            total_reward += self.reward
            self.test_renderer.render_board(self.display)
            pygame.display.update()

            # Lower than usual frame rate - will slow down disappearing line animations unnaturally,
            # but should allow a human to follow what the computer's doing in real time!
            self.clock.tick(10)

        print( "Final reward: %.1f"%total_reward )
        print( "Saving to file %s ..."%path )
        self.save(path)

    #-------------------------------------------------------------------------------------------

    def save(self, path):
        self.experience_buffer.save(path, True)

    def load(self,path):
        self.experience_buffer.load(path)


if __name__ == '__main__':
    game = ExperienceBufferBuilder()

    game.prepare_run()
    game.run('pgbuffer1.bf')
