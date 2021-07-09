import numpy as np

from . import PieceType

class Piece:
    def __init__(self, type):
        self.type = type
        self.mask = np.copy(self.type.mask)

def random(piece_bag = []):
    type, piece_bag = PieceType.select_random(piece_bag)
    return (Piece(type), piece_bag)
