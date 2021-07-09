from . import Actions
from .Board import Board

MAX_DIFFICULTY = 20

# Difficulty automatically raised after this many frames
DIFFICULTY_RAISE_TIME = 60*60


class Game:
    def __init__(self, dynamic_difficulty=True, width=10, height=20):
        self.board = Board(self, width, height)
        self.dynamic_difficulty = dynamic_difficulty
        self.difficulty = 0

    def start(self):
        self.frame_count = 0
        self.game_complete = False
        self.board.start()

    def notify_score(self,score):
        pass

    def notify_piece_landed(self):
        pass

    def notify_game_over(self):
        self.game_complete = True

    def handle_action(self,action):
        if action == Actions.QuitGame:
            self.game_complete = True
        else:
            self.board.handle_action(action)


    def update(self, actions):
        for action in actions:
            self.handle_action(action)

        self.board.update()

        self.frame_count += 1

        if self.dynamic_difficulty:
            if self.frame_count%DIFFICULTY_RAISE_TIME==0 and self.difficulty<MAX_DIFFICULTY:
                self.difficulty += 1
