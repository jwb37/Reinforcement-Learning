import pygame

from . import Actions
from .Game import Game
from .Board import Board
from .PygameRenderer import PygameRenderer


# Game runs at fixed fps (unless training AI)
FRAME_RATE = 60.0
#MAX_FRAME_TIME = 1.0/FRAME_RATE


class InteractiveGame(Game):
    def __init__(self):
        super().__init__()

        pygame.init()

        width, height = 400, 880

        self.display = pygame.display.set_mode( (width, height) )
        pygame.display.set_caption('Tetris')

        self.clock = pygame.time.Clock()

        self.renderer = PygameRenderer(self)
        self.renderer.resize( width, height )

    def start(self):
        super().start()

        self.difficulty = 20

        while not self.game_complete:
            actions = []

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    actions.append(Actions.QuitGame)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        actions.append(Actions.Left)
                    elif event.key == pygame.K_RIGHT:
                        actions.append(Actions.Right)
                    elif event.key == pygame.K_DOWN:
                        actions.append(Actions.FastFallOn)
                    elif event.key == pygame.K_z:
                        actions.append(Actions.RotateLeft)
                    elif event.key == pygame.K_x:
                        actions.append(Actions.RotateRight)
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_DOWN:
                        actions.append(Actions.FastFallOff)

            self.update(actions)

            self.renderer.render_board(self.display)
            pygame.display.update()

            self.clock.tick(FRAME_RATE)
        self.end()

    def end(self):
        pygame.quit()
        print("Game over")
        print("Final score: %d"%(self.board.score))
        quit()


if __name__ == "__main__":
    game = InteractiveGame()
    game.start()
