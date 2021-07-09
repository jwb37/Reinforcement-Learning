import random
import numpy as np



#--------------------------------------------------------------------------------------------------
# Rotation Helper Function
#--------------------------------------------------------------------------------------------------
def _rh(n, coord_tuple_array):
    coord_tuple = coord_tuple_array[n]
    if (n % 2):
        return (coord_tuple[0], coord_tuple[1], Ellipsis, coord_tuple[2])
    else:
        return (coord_tuple[0], coord_tuple[1], coord_tuple[2], Ellipsis)



class Cube:

    actions = [ (axis, lr, k) for axis in range(3) for lr in range(2) for k in (True,False) ]

    def __init__(self, side_length):
        # Blocks stored in tensor of form x/y/z axis, left/right, row (top->bottom) then column(l->r)

        self.side_length = side_length
        self.blocks = np.zeros( (3, 2, side_length, side_length), dtype=np.byte )
        self.reset()

    def reset(self):
        # 'Colour' each face a solid colour.
        for axis in range(3):
            for lr in range(2):
                self.blocks[ axis, lr, :, : ] = 2*axis + lr

    #------------------------------------------------------------------------------------------------------------

    def rotate_one_step(self, axis, lr, anticlockwise=True):
        # Took quite some pen and paper work to figure this out!
        # Axes follow a left-hand-rule orientation
        # Each matrix [ax, lr, :, :] is the face seen looking at the cube along the axis, with the
        # 'up' direction being in the direction of the face given by [(ax+1)%3, r, :, :]
        a1 = (axis+1)%3
        a2 = (a1+1)%3
        n = self.side_length - 1

        if lr == 0:
            array = (
                (a1, 1, n),
                (a2, 0, 0),
                (a1, 0, 0),
                (a2, 1, 0)
            )
        elif lr == 1:
            array = (
                (a1, 1, 0),
                (a2, 1, n),
                (a1, 0, n),
                (a2, 0, n)
            )

        if anticlockwise:
            tmp_line = self.blocks[_rh(3, array)].copy()
            for n in range(3,0,-1):
                self.blocks[_rh(n, array)] = self.blocks[_rh(n-1, array)].copy()
            self.blocks[_rh(0, array)] = tmp_line
        else:
            tmp_line = self.blocks[_rh(0, array)].copy()
            for n in range(0,3):
                self.blocks[_rh(n, array)] = self.blocks[_rh(n+1, array)].copy()
            self.blocks[_rh(3, array)] = tmp_line

        if anticlockwise:
            self.blocks[axis, lr] = np.rot90( self.blocks[axis, lr], 1 )
        else:
            self.blocks[axis, lr] = np.rot90( self.blocks[axis, lr], -1 )

    #------------------------------------------------------------------------------------------------------------

    def take_action(self, action):
        self.rotate_one_step( *action )

    def take_action_by_idx(self, idx):
        self.take_action( self.actions[idx] )

    def check_solved(self):
        solved = True

        for axis in range(3):
            for lr in range(2):
                if not (self.blocks[axis, lr, 0, 0] == self.blocks[axis, lr]).all():
                    solved = False

        return solved

    def scramble(self, num_rotations, return_scramble_steps=False):
        if return_scramble_steps:
            actions = []

        for a in range(num_rotations):
            action = random.choice(self.actions)
            self.take_action(action)
            if return_scramble_steps:
                actions.append(action)

        if return_scramble_steps:
            return actions

    @property
    def state(self):
        return self.blocks.flatten()
