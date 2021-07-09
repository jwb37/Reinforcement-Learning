import numpy as np

from Board import Board
from BlockState import BlockState
from BlockColour import BlockColour


board = Board(None,6,12)

board.block_states[:4,:] = BlockState.RESTING
board.block_states[4:,:] = BlockState.NONE

board.block_colours[:] = np.random.randint(3,6, (6,12))

def print_array(arr):
    print(arr.T[::-1,:])


print("States")
print_array(board.block_states)
print("Colours")
print_array(board.block_colours)
print("Matches")
print_array(board.check_matches())

needs_updating = np.full_like( board.block_colours, True )
blocks_to_explode = board.check_matches()
needs_updating = np.logical_and(needs_updating, np.logical_not(blocks_to_explode))
board.block_states[blocks_to_explode] = BlockState.EXPLODING

print("States")
print_array(board.block_states)
print("Needs Updating")
print_array(needs_updating)
