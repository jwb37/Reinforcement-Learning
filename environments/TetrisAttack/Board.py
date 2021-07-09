import numpy as np
import random

from . import Actions
from . import BlockAnimations
from .BlockState import BlockState
from .BlockColour import BlockColour


#------------------------------------------------------------------------------------------
# Utility functions
# Shift all rows/columns in the matrix in the specified direction. One row/column will get scrolled
# out of existence, and at the opposite end of the matrix, the empty space is filled with 'fill_value'
#------------------------------------------------------------------------------------------
def left_shift(array, fill_value):
    answer = np.empty_like(array)
    answer[ :-1, : ] = array[ 1:, : ]
    answer[ -1, : ] = fill_value
    return answer

def right_shift(array, fill_value):
    answer = np.empty_like(array)
    answer[ 1:, : ] = array[ :-1, : ]
    answer[ 0, : ] = fill_value
    return answer

def down_shift(array, fill_value):
    answer = np.empty_like(array)
    answer[ :, :-1 ] = array[ :, 1: ]
    answer[ :, -1 ] = fill_value
    return answer

def up_shift(array, fill_value):
    answer = np.empty_like(array)
    answer[ :, 1: ] = array[ :, :-1 ]
    answer[ :, 0 ] = fill_value
    return answer

#------------------------------------------------------------------------------------------
# Debug function
#------------------------------------------------------------------------------------------
def print_array(arr):
    print(arr.T[::-1,:].astype('int'))


class Board:

    def __init__(self, game, width=6, height=12):
        self.width = width
        self.height = height

        self.game = game

    def start(self):
        self.block_colours = np.full( (self.width, self.height), BlockColour.NONE )
        self.block_offsets = np.zeros( (self.width, self.height) )
        self.block_states = np.full( (self.width, self.height), BlockState.NONE )
        self.block_animation_start_times = np.zeros( (self.width, self.height), np.single )

        self.populate_random()

        self.generate_new_row()

        self.fastscroll = False
        self.scroll_offset = 0.0
        self.scroll_paused = False
        self.scroll_paused_time = 0.0
        self.next_scroll_time = 2
        self.scroll_increment = 0.1
        self.fall_speed = 0.5

        self.game_over = False

        self.cursor_x = max(min(self.width-2, int(self.width/2)),0)
        self.cursor_y = max(min(self.height-1, int(self.height/2)),0)

    def set_game_over(self, game_over_state):
        self.game_over = game_over_state
        if self.game_over:
            self.game.notify_game_over()

    def check_game_over(self):
        return self.game_over

    #------------------------------------------------------------------------------------------
    def populate_random(self):
        for x in range(self.width):
            height = random.randint(3,6)
            for y in range(height):
                self.block_colours[x,y] = BlockColour.random()
                self.block_states[x,y] = BlockState.RESTING
    #------------------------------------------------------------------------------------------
    def check_matches(self):
        '''check_matches - Finds all blocks in the board that are part of either 3 (or more) in a row vertically
                            or 3 (or more) in a row horizontally.
           Returns a boolean array of the same shape as the board, with True for those blocks which are part of a matching set'''
           
        colours_shifted = left_shift(self.block_colours, BlockColour.NONE)
        # match2 reads True in any position where the block to the right matches the colour of the current block
        match2 = (self.block_colours == colours_shifted)
        # match3 reads True in any position for which the next 3 in the row match the colour
        match3 = np.logical_and(match2, left_shift(match2, False))
        # Ensure we only match blocks which are resting
        horizontal_matches = np.logical_and(match3, self.block_states == BlockState.RESTING)
        # After the next bit of code, horizontal_matches contains exactly those blocks that are in a 3 (or more) in a row horizontally
        for i in range(0,2):
            horizontal_matches = np.logical_or(horizontal_matches, right_shift(horizontal_matches, False))

        # Repat the above code with suitable changes to check for vertical matches
        colours_shifted = down_shift(self.block_colours, BlockColour.NONE)
        match2 = (self.block_colours == colours_shifted)
        match3 = np.logical_and(match2, down_shift(match2, False))
        vertical_matches = np.logical_and(match3, self.block_states == BlockState.RESTING)
        for i in range(0,2):
            vertical_matches = np.logical_or(vertical_matches, up_shift(vertical_matches, False))

        self.check_matches_needed = False
        return np.logical_or(horizontal_matches, vertical_matches)

    #------------------------------------------------------------------------------------------
    def remove_block(self, x, y):
        self.block_states[x,y] = BlockState.NONE
        self.block_colours[x,y] = BlockColour.NONE
        self.block_animation_start_times[x,y] = 0.0
        self.block_offsets[x,y] = 0

    #------------------------------------------------------------------------------------------
    def update(self):

        if self.game_over:
            return

        needs_updating = np.full_like( self.block_colours, True )
        current_time = self.game.frame_count

        # check for resting blocks that need to explode
        blocks_to_explode = self.check_matches()
        needs_updating = np.logical_and(needs_updating, np.logical_not(blocks_to_explode))
        self.block_states[blocks_to_explode] = BlockState.EXPLODING
        self.block_animation_start_times[blocks_to_explode] = current_time

        # Scoring needs to be made more advanced (count chains etc)
        # For now, just inform of the number of blocks exploded
        self.game.notify_score(np.count_nonzero(blocks_to_explode))

        # Loop through all blocks
        for x in range(0, self.width):
            # Note: The falling logic below REQUIRES this y loop to run from smallest to largest
            for y in range(0, self.height):

                    if not needs_updating[x,y]:
                        continue

                    animation_time = current_time - self.block_animation_start_times[x,y]

                    # check for exploding animations running out
                    if self.block_states[x,y] == BlockState.EXPLODING:
                        needs_updating[x,y] = False
                        if animation_time > BlockAnimations.EXPLODE_TIME:
                            self.remove_block(x,y)
                        continue

                    # check for floating animations running out
                    elif self.block_states[x,y] == BlockState.FLOATING:
                        needs_updating[x,y] = False
                        if animation_time > BlockAnimations.FLOAT_TIME:
                            # Timer has run out. Set falling state
                            self.block_states[x,y] = BlockState.FALLING
                            self.block_animation_start_times[x,y] = current_time
                        continue
                            
                    # check for switching animations running out
                    # We only look for the block which is switching right and update both blocks at once
                    # Assumptions made (following game logic):
                    #       block switching right cannot be in far right column of board
                    #       block switching left will be immediately to the right of block switching right
                    #       blocks have equal animation start times
                    elif self.block_states[x,y] == BlockState.SWITCHING_RIGHT:
                        assert x+1 < self.width, "Block switching right is trying to move off the far end of the board"
                        assert self.block_states[x+1,y] == BlockState.SWITCHING_LEFT, "Block switching left is not to the right of the block switching right"

                        needs_updating[x,y] = False
                        needs_updating[x+1,y] = False
                        if animation_time > BlockAnimations.SWITCH_TIME:
                            # Timer has run out. Switch positions and set resting
                            self.block_colours[x,y], self.block_colours[x+1,y] = self.block_colours[x+1,y], self.block_colours[x,y]
                            self.block_offsets[x,y] = 0.0
                            self.block_offsets[x+1,y] = 0.0

                            if self.block_colours[x,y] == BlockColour.NONE:
                                self.block_states[x,y] = BlockState.NONE
                            else:
                                self.block_states[x,y] = BlockState.RESTING

                            if self.block_colours[x+1,y] == BlockColour.NONE:
                                self.block_states[x+1,y] = BlockState.NONE
                            else:
                                self.block_states[x+1,y] = BlockState.RESTING
                        else:
                            self.block_offsets[x,y] = float(animation_time)/BlockAnimations.SWITCH_TIME
                            self.block_offsets[x+1,y] = -float(animation_time)/BlockAnimations.SWITCH_TIME

                    # Update offsets of falling blocks and check for landing
                    elif self.block_states[x,y] == BlockState.FALLING:
                        needs_updating[x,y] = False
                        # Note: logic below only holds up if:
                        #       fall_speed < 1.0
                        #       the outer loop runs through y coordinates from smallest to largest

                        # Check if should land on block below
                        if y == 0 or not self.block_states[x,y-1] in (BlockState.NONE, BlockState.FALLING):
                            self.block_states[x,y] = BlockState.RESTING
                            self.block_offsets[x,y] = 0.0
                        else:
                            self.block_offsets[x,y] -= self.fall_speed
                            if self.block_offsets[x,y] < -1.0:
                                self.block_offsets[x,y] += 1.0

                                self.block_states[x,y-1] = self.block_states[x,y]
                                self.block_colours[x,y-1] = self.block_colours[x,y]
                                self.block_offsets[x,y-1] = self.block_offsets[x,y]

                                self.block_states[x,y] = BlockState.NONE
                                self.block_colours[x,y] = BlockColour.NONE
                                self.block_offsets[x,y] = 0.0
                    
                    # check for resting blocks that need to float
                    elif self.block_states[x,y] == BlockState.RESTING:
                        needs_updating[x,y] = False
                        if y > 0 and self.block_states[x,y-1] in (BlockState.NONE, BlockState.FALLING):
                            self.block_states[x,y] = BlockState.FLOATING
                            self.block_animation_start_times[x,y] = current_time
                        continue


        # Disallow scrolling if any blocks are exploding, falling or floating.
        # It would be more performant to check for this in the above loop
        # but this is much nicer to read! :)
        # Maybe when training an AI, switch to the more performant version
        disallow_scroll = (
            (self.block_states == BlockState.FALLING).any()
            or (self.block_states == BlockState.EXPLODING).any()
            or (self.block_states == BlockState.FLOATING).any()
        )
        self.set_scroll_state(not disallow_scroll)

        if not self.scroll_paused and current_time > self.next_scroll_time:
            self.increment_scroll()


    #------------------------------------------------------------------------------------------
    # Functions related to scrolling of the board
    #------------------------------------------------------------------------------------------
    def pause_scroll(self):
        if self.scroll_paused:
            return
        self.scroll_paused = True
        self.scroll_paused_time = self.game.frame_count


    def unpause_scroll(self):
        if not self.scroll_paused:
            return
        self.next_scroll_time += self.game.frame_count - self.scroll_paused_time
        self.scroll_paused = False


    def set_scroll_state(self, allow_scroll):
        if allow_scroll and self.scroll_paused:
            self.unpause_scroll()
        if not allow_scroll and not self.scroll_paused:
            self.pause_scroll()

    def set_next_scroll_time(self):
        if self.fastscroll:
            scroll_time = 1
        else:
            scroll_time = max( 10 - self.game.frame_count/(20*60), 1 )

        self.next_scroll_time = self.game.frame_count + scroll_time

    def increment_scroll(self):
        self.scroll_offset += self.scroll_increment

        # If scroll_offset exceeds 1, we need to shift all the blocks up a space in the grid
        # and fill the bottom_space with the next row.
        # At time of writing, I don't foresee a scrolling speed high enough that it should
        # ever exceed a whole block in a single frame, but just in case, I'm accounting for
        # that case here anyway
        while self.scroll_offset > 1.0:

            # If there are any blocks in the top row, trigger a game over
            if (self.block_states[ :, -1 ] != BlockState.NONE).any():
                self.set_game_over(True)
                return

            self.block_colours = up_shift(self.block_colours, 0)
            self.block_states = up_shift(self.block_states, BlockState.RESTING)
            self.block_animation_start_times = up_shift(self.block_animation_start_times, self.game.frame_count)
            self.block_offsets = up_shift(self.block_offsets, 0.0)

            self.cursor_y = min(self.cursor_y+1, self.height)

            # Promote the 'new_row' into the bottom row of the grid proper and prepare the next 'new_row'
            self.block_colours[ :, 0 ] = self.new_row_colours[ : ]
            self.generate_new_row()

            self.scroll_offset -= 1.0

        self.set_next_scroll_time()


    def generate_new_row(self):
        new_row = []

        # Do the last two consecutive blocks have the same colour?
        two_in_row = False

        for n in range(self.width):
            new_colour = BlockColour.random()

            # Logic to ensure that we never generate 3 blocks in a row with the same colour
            if n > 0:
                if two_in_row:
                    while new_colour == new_row[n-1]:
                        new_colour = BlockColour.random()
                else:
                    if new_colour == new_row[n-1]:
                        two_in_row = True

            new_row.append(new_colour)
        
        self.new_row_colours = np.array(new_row)


    #------------------------------------------------------------------------------------------
    def handle_action(self, action):

        if action == Actions.Left:
            self.cursor_x = max(0, self.cursor_x-1)

        elif action == Actions.Right:
            self.cursor_x = min(self.width - 2, self.cursor_x+1)

        elif action == Actions.Up:
            self.cursor_y = min(self.height - 2, self.cursor_y+1)

        elif action == Actions.Down:
            self.cursor_y = max(0, self.cursor_y-1)

        elif action == Actions.Switch:
            state_left, state_right = self.block_states[self.cursor_x:self.cursor_x+2, self.cursor_y]
            if not state_left in {BlockState.RESTING, BlockState.NONE, BlockState.FLOATING}:
                return
            if not state_right in {BlockState.RESTING, BlockState.NONE, BlockState.FLOATING}:
                return

            current_time = self.game.frame_count

            self.block_states[self.cursor_x, self.cursor_y] = BlockState.SWITCHING_RIGHT
            self.block_animation_start_times[self.cursor_x, self.cursor_y] = current_time
            self.block_offsets[self.cursor_x, self.cursor_y] = 0.0

            self.block_states[self.cursor_x+1, self.cursor_y] = BlockState.SWITCHING_LEFT
            self.block_animation_start_times[self.cursor_x+1, self.cursor_y] = current_time
            self.block_offsets[self.cursor_x+1, self.cursor_y] = 0.0

        elif action == Actions.FastScroll_On:
            self.fastscroll = True

        elif action == Actions.FastScroll_Off:
            self.fastscroll = False
