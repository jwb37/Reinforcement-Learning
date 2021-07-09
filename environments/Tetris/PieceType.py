import random
import numpy as np

class PieceType:
    def __init__(self,mask,name=None):
        self.mask = mask
        self.name = name

piece_types = [
    PieceType(np.array([
        [ 0, 0, 0, 0 ],
        [ 0, 1, 1, 0 ],
        [ 0, 1, 1, 0 ],
        [ 0, 0, 0, 0 ]
    ]), "Square"),
    PieceType(np.array([
        [ 0, 0, 1, 0 ],
        [ 0, 0, 1, 0 ],
        [ 0, 0, 1, 0 ],
        [ 0, 0, 1, 0 ]
    ]), "Straight"),
    PieceType(np.array([
        [ 0, 0, 0, 0 ],
        [ 0, 0, 1, 0 ],
        [ 0, 1, 1, 1 ],
        [ 0, 0, 0, 0 ]
    ]), "T-Shape"),
    PieceType(np.array([
        [ 0, 0, 0, 0 ],
        [ 1, 1, 0, 0 ],
        [ 0, 1, 1, 0 ],
        [ 0, 0, 0, 0 ]
    ]), "Z Shape L"),
    PieceType(np.array([
        [ 0, 0, 0, 0 ],
        [ 0, 0, 1, 1 ],
        [ 0, 1, 1, 0 ],
        [ 0, 0, 0, 0 ]
    ]), "Z Shape R"),
    PieceType(np.array([
        [ 0, 0, 0, 0 ],
        [ 0, 1, 1, 0 ],
        [ 0, 0, 1, 0 ],
        [ 0, 0, 1, 0 ]
    ]), "L Shape L"), 
    PieceType(np.array([
        [ 0, 0, 0, 0 ],
        [ 0, 1, 1, 0 ],
        [ 0, 1, 0, 0 ],
        [ 0, 1, 0, 0 ]
    ]), "L Shape R")
]


# Every piece comes up twice in every successive 14 pieces, but in a randomly determined order
def select_random(piece_bag=[]):
    if not piece_bag:
        piece_bag = piece_types.copy()

    chosen_type = random.choice(piece_bag)
    piece_bag.remove(chosen_type)

    return (chosen_type, piece_bag)
