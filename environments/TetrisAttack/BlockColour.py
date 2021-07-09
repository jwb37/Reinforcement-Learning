#from enum import Enum
import random

NUM_COLOURS = 5

class BlockColour:
    NONE=0
    RUBBISH=1
    SPECIAL=2
    def random():
        return random.randint(2, 2+NUM_COLOURS)
