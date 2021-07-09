import math
import numpy as np

from . import Piece
from . import BoardState
from . import Actions
from . import AnimationTimings


PIECE_QUEUE_SIZE = 2


class Board:

    def __init__(self, game, width=10, height=20):
        # Note: Walls and floor now permanently built into blocks array
        # Helps considerably with checking collision
        self.blocks = np.zeros( (width+4,height+2), dtype='b' )
        self.state = BoardState.Paused
        self.game = game


    def start(self):
        self.state = BoardState.Normal
        self.animation_start = 0
        self.score = 0
        self.fastfall = False
        self.game_over = False

        self.blocks[ : ] = False
        self.blocks[ :2, : ] = True
        self.blocks[ -2:, : ] = True
        self.blocks[ : , :2] = True

        self.piece = None
        self.piece_queue = []

        self.piece_bag = []
        for i in range(PIECE_QUEUE_SIZE):
            new_piece, self.piece_bag = Piece.random(self.piece_bag)
            self.piece_queue.append( new_piece )

        self.spawn_piece()

    @property
    def width(self):
        return self.blocks.shape[0]

    @property
    def height(self):
        return self.blocks.shape[1]

    def piece_falling_delay(self):
        if self.fastfall:
            delay = AnimationTimings.MinPieceFallDelay
        else:
            # delay is 60 frames at difficulty 0 and 20 frames at difficulty 20 and scales linearly.
            # => Reaches 1 frame per second at difficulty 29.5
            delay = 60*(20 - self.game.difficulty)/20.0 + 20*(self.game.difficulty)/20
            #5*math.exp(-self.game.difficulty/10.0) + 1

        return max(delay, AnimationTimings.MinPieceFallDelay)

    def report_score(self, number_of_lines_cleared):
        points_scored = 10 * (2**(number_of_lines_cleared-1))

        self.game.notify_score( points_scored )
        self.score += points_scored

    def report_game_over(self):
        self.state = BoardState.GameOver
        self.game.notify_game_over()

    # ---------------------------------------------------------------------------------    
    # Game logic and update functions

    def spawn_piece(self):
        self.piece = self.piece_queue.pop()
        self.piece.x = int(self.width / 2)
        self.piece.y = self.height - 2

        # This is a fairly abrupt game over. Make this smoother and fairer?
        if self.check_collision( self.piece.mask, self.piece.x, self.piece.y ):
            self.report_game_over()
        
        new_piece, self.piece_bag = Piece.random(self.piece_bag)
        self.piece_queue.append( new_piece )

    def check_lines(self):
        # Note: Important for update function that lines_to_clear is sorted from lowest to highest line
        
        lines_to_clear = []

        for y in range(2,self.height):
            if self.blocks[ 2:-2, y ].all():
                lines_to_clear.append(y)

        if lines_to_clear:
            self.state = BoardState.Clearing
            self.animation_start = self.game.frame_count
            self.lines_to_clear = lines_to_clear
            self.report_score(len(lines_to_clear))
            return True
        else:
            return False

    def check_collision(self, mask, x, y):
        ''' Checks whether a 4x4 mask would collide with the board if the
            centre of the mask is placed at position (x,y).
            Checks for collisions both with blocks already placed on the board
            and with the sides/bottom of the board'''

        if x<2 or x>self.width-2 or y<2 or y>self.height-2:
            return True

        full_mask = np.zeros_like(self.blocks)
        full_mask[ x-2:x+2, y-2:y+2 ] = mask

        return np.logical_and(self.blocks, full_mask).any()

    def check_piece_landed(self):
        '''Works by seeing what squares the piece would cover after the next fall,
           and seeing if they intersect with the blocks already on the board.
           Because of the design of check_collision, also checks if piece has landed
           at the bottom of the board'''

        return self.check_collision( self.piece.mask, self.piece.x, self.piece.y - 1 )

    def update(self):
        if self.state == BoardState.Paused:
            return

        delta_frames = self.game.frame_count - self.animation_start

        if self.state == BoardState.Clearing:
            if delta_frames < AnimationTimings.ClearingTime:
                return
            else:
                self.state = BoardState.Falling
                self.animation_start = self.game.frame_count
                return

        if self.state == BoardState.Falling:
            if delta_frames > 0 and delta_frames % AnimationTimings.LineFallingDelay == 0:
                if not self.lines_to_clear:
                    self.state = BoardState.Normal
                    self.animation_start = 0
                    self.spawn_piece()
                    return

                for y in range(self.lines_to_clear[0], self.height-1):
                    self.blocks[ 2:-2, y ] = self.blocks[ 2:-2, y+1 ]
                self.blocks[ 2:-2, -1 ] = 0

                self.lines_to_clear.pop(0)
                for idx in range(len(self.lines_to_clear)):
                    self.lines_to_clear[idx] -= 1

                return
            else:
                return

        if self.state == BoardState.Normal:
            if delta_frames > 0 and delta_frames % self.piece_falling_delay() == 0:
                if self.check_piece_landed():
                    mask = np.zeros_like(self.blocks)
                    mask[ self.piece.x-2 : self.piece.x+2, self.piece.y-2 : self.piece.y+2 ] = self.piece.mask
                    self.blocks = np.logical_or(self.blocks, mask)

                    self.game.notify_piece_landed()

                    if not self.check_lines():
                        self.state = BoardState.Normal
                        self.animation_start = 0
                        self.spawn_piece()
                    return
                else:
                    self.piece.y -= 1
                    return
            else:
                return

    def handle_action(self, action):
        if action == Actions.Left:
            if not self.check_collision(self.piece.mask, self.piece.x-1, self.piece.y):
                self.piece.x -= 1
        elif action == Actions.Right:
            if not self.check_collision(self.piece.mask, self.piece.x+1, self.piece.y):
                self.piece.x += 1
        elif action == Actions.RotateLeft:
            rotated_mask = np.rot90(self.piece.mask, k=-1)
            if not self.check_collision(rotated_mask, self.piece.x, self.piece.y):
                self.piece.mask = rotated_mask
        elif action == Actions.RotateRight:
            rotated_mask = np.rot90(self.piece.mask, k=1)
            if not self.check_collision(rotated_mask, self.piece.x, self.piece.y):
                self.piece.mask = rotated_mask
        elif action == Actions.FastFallOn:
            self.fastfall = True
        elif action == Actions.FastFallOff:
            self.fastfall = False


    # ---------------------------------------------------------------------------------
    # Board analytics functions
    #   For potential use in reward functions

    def max_heights(self):
        ''' Returns an array with an entry for each column of the board.
            Each entry represents the height of the highest block in that column
        '''
        max_heights = np.zeros( (self.width), dtype=int )

        for x in range(self.width):
            heights_of_blocks = self.blocks[x,:].nonzero()[0]
            if heights_of_blocks.any():
                max_heights[x] = max(heights_of_blocks)

        return max_heights


    def max_height(self):
        ''' Returns the height of the highest block on the board
        '''
        return max(self.max_heights()[2:self.width-2])


    def count_holes(self):
        '''Counts the number of 'holes' in the board
           A 'hole' is any empty space with a filled space above it.
           Potentially to be used in the reward function for a ML trained model
        '''
        holes = 0

        max_heights = self.max_heights()

        for x in range(self.width):
            holes += max_heights[x] - np.count_nonzero(self.blocks[x,:max_heights[x]])

        return holes
