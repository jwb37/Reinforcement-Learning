import pygame

from .Board import Board
from .PygameRenderer import PygameRenderer
from . import Actions

FPS = 60

class InteractiveGame:

    RewardInfo = {
        "GameOver": -20,
        "ScoreExponent": 1.3,
        "ScoreFactor": 1
    }

    def __init__(self, num_players=1, difficulty=0):
        self.board = Board(self, width=6, height=12)

        pygame.init()

        width, height = 300, 600
        #width, height = 60, 120

        self.display = pygame.display.set_mode( (width, height) )
        pygame.display.set_caption('Tetris Attack')

        self.clock = pygame.time.Clock()

        self.renderer = PygameRenderer(self)
        self.renderer.resize( width, height )

        self.paused = False
        self.finished = False

    def notify_game_over(self):
        self.finished = True
        self.reward += self.RewardInfo["GameOver"]

    def notify_score(self, blocks_cleared):
        self.reward += self.RewardInfo["ScoreFactor"] * (blocks_cleared ** self.RewardInfo["ScoreExponent"])

    def read_events(self):
        actions = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                actions.append(Actions.QuitGame)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    actions.append(Actions.Left)
                elif event.key == pygame.K_RIGHT:
                    actions.append(Actions.Right)
                elif event.key == pygame.K_UP:
                    actions.append(Actions.Up)
                elif event.key == pygame.K_DOWN:
                    actions.append(Actions.Down)
                elif event.key == pygame.K_c:
                    actions.append(Actions.FastScroll_On)
                elif event.key == pygame.K_x:
                    actions.append(Actions.Switch)
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_c:
                    actions.append(Actions.FastScroll_Off)
        return actions

    def handle_actions(self, actions):
        for action in actions:
            if action == Actions.QuitGame:
                self.finished = True
            else:
                self.board.handle_action(action)

    def start(self):
        self.board.start()
        self.frame_count = 0

        # Main game loop
        while not self.finished:
            actions = self.read_events()

            self.handle_actions(actions)

            self.reward = 0
            self.board.update()
            if self.reward > 0:
                print( f"Reward: {self.reward}" )

            self.renderer.render_board( self.board, self.display )
            pygame.display.update()

            self.clock.tick(FPS)
            self.frame_count += 1

        self.end()

    def end(self):
        pygame.quit()
        print("Game over")
        quit()


if __name__ == "__main__":
    game = InteractiveGame()
    game.start()
