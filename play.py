import sys
from environments import load_interactive_env

game_name = sys.argv[1]

if __name__ == '__main__':
    game = load_interactive_env(game_name)
    game.start()
