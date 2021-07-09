import pygame
import numpy as np

from .BlockState import BlockState
from .BlockColour import BlockColour

shapes = [
    # Special block
    {
        "name": "star",
        "vertices": [
            (0.742705098312484, 0.676335575687742),
            (0.654508497187474, 0.975528258147577),
            (0.5-0.092705098312484, 0.785316954888546),
            (0.5-0.404508497187474, 0.793892626146237),
            (0.5-0.3, 0.5),
            (0.5-0.404508497187474, 0.5-0.293892626146236),
            (0.5-0.092705098312484, 0.5-0.285316954888546),
            (0.654508497187474, 0.5-0.475528258147577),
            (0.742705098312484, 0.5-0.176335575687742),
            (1.0, 0.5)
        ],
        "bg_colour": (70, 70, 70),
        "shape_colour": (180, 180, 180),
        "outline_colour": (20, 20, 20)
    },
    # Coloured blocks
    {
        "name": "triangle",
        "vertices": [ (0.25, 0.3), (0.75, 0.3), (0.5, 0.733013) ],
        "bg_colour": (70, 255, 70),
        "shape_colour": (180, 255, 180),
        "outline_colour": (0, 80, 0)
    },
    {
        "name": "square",
        "vertices": [ (0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.2) ],
        "bg_colour": (150, 0, 150),
        "shape_colour": (255, 180, 255),
        "outline_colour": (80, 0, 80)
    },
    {
        "name": "diamond",
        "vertices": [ (0.2, 0.5), (0.5, 0.2), (0.8, 0.5), (0.5, 0.8) ],
        "bg_colour": (70, 70, 255),
        "shape_colour": (180, 180, 255),
        "outline_colour": (0, 0, 80)
    },
    {
        "name": "cross",
        "vertices": [ (0.4, 0.2), (0.6, 0.2), (0.6, 0.4), (0.8, 0.4), (0.8, 0.6), (0.6, 0.6), (0.6, 0.8), (0.4, 0.8), (0.4, 0.6), (0.2, 0.6), (0.2, 0.4), (0.4, 0.4) ],
        "bg_colour": (0, 150, 150),
        "shape_colour": (180, 255, 255),
        "outline_colour": (0, 80, 80)
    },
    {
        "name": "pentagon",
        "vertices": [ (0.785317, 0.592705), (0.5, 0.8), (0.5-0.285317, 0.592705), (0.5-0.176336, 0.5-0.242705), (0.676336, 0.5-0.242705) ],
        "bg_colour": (150, 150, 0),
        "shape_colour": (255, 255, 180),
        "outline_colour": (80, 80, 0)
    }
]



class PygameRenderer:
    def __init__(self, game):
        self.game = game


    def resize(self, screen_width, screen_height):
        grid_width = self.game.board.width
        grid_height = self.game.board.height

        self.block_width = screen_width/grid_width
        self.block_height = screen_height/grid_height

        self.create_sprites( self.block_width, self.block_height )


    def create_sprites(self, width, height):
        self.block_sprites = []

        self.cursor_sprite = pygame.Surface( (width*2 + 2, height + 2), pygame.SRCALPHA )
        self.cursor_sprite.fill( (0, 0, 0, 0) )

        pygame.draw.lines(
            self.cursor_sprite,
            (255, 255, 255, 255),
            True,
            [ (1,1), (width,1), (width*2+1,1), (width*2+1, height), (width, height), (1, height)],
            3
        )
        pygame.draw.line(
            self.cursor_sprite,
            (255, 255, 255, 255),
            (width,1),
            (width, height),
            3
        )

        exploding_sprite = pygame.Surface( (width, height) )
        # TODO: Proper graphics/animation here!
        exploding_sprite.fill( (255,0,0) )

        pygame.draw.aalines(
            exploding_sprite,
            (0,0,0),
            True,
            [ (0,0), (width-1,0), (width-1,height-1), (0,height-1) ]
        )

        for shape in shapes:
            sprite = pygame.Surface( (width, height) )
            sprite.fill( shape["bg_colour"] )

            vertices = np.array( shape["vertices"] )
            vertices[ :, 0 ] *= width
            vertices[ :, 1 ] *= height

            pygame.draw.polygon(sprite, shape["shape_colour"], vertices)
            pygame.draw.polygon(sprite, shape["outline_colour"], vertices, 2)

            pygame.draw.aalines(
                sprite,
                (0,0,0),
                True,
                [ (0,0), (width-1,0), (width-1,height-1), (0,height-1) ]
            )

            greyed_sprite = sprite.copy()
            grey_square = pygame.Surface( (width, height) )
            grey_square.fill( (90, 90, 90) )
            greyed_sprite.blit( grey_square, (0, 0), special_flags=pygame.BLEND_MULT )

            self.block_sprites.append({
                BlockState.RESTING: sprite,
                BlockState.SWITCHING_LEFT: sprite,
                BlockState.SWITCHING_RIGHT: sprite,
                BlockState.FLOATING: sprite,
                BlockState.FALLING: sprite,
                BlockState.APPEARING: greyed_sprite,
                BlockState.EXPLODING: exploding_sprite
            })


    def render_board(self, board, render_target):
        screen_width = render_target.get_width()
        screen_height = render_target.get_height()

        render_target.fill( (180, 180, 255) )

        for x in range(board.width):
            #Draw main board
            for y in range(board.height):
                blockstate = board.block_states[x,y]

                if blockstate == BlockState.NONE:
                    continue

                block_colour = board.block_colours[x,y]
                if block_colour == BlockColour.NONE:
                    continue

                grid_x = x
                grid_y = y + board.scroll_offset

                if blockstate in (BlockState.SWITCHING_LEFT, BlockState.SWITCHING_RIGHT):
                    grid_x += board.block_offsets[x,y]
                elif blockstate == BlockState.FALLING:
                    grid_y += board.block_offsets[x,y]

                screen_x = grid_x * self.block_width
                screen_y = screen_height - (grid_y+1) * self.block_height

                render_target.blit(
                    self.block_sprites[block_colour - 2][blockstate],
                    (screen_x, screen_y)
                )

            # Draw greyed out new row sprites
            block_colour = board.new_row_colours[x]
            render_target.blit(
                self.block_sprites[block_colour - 2][BlockState.APPEARING],
                (x * self.block_width, screen_height - board.scroll_offset*self.block_height)
            )

        # Draw cursor
        screen_x = board.cursor_x * self.block_width + 1
        screen_y = screen_height - (board.cursor_y+1+board.scroll_offset) * self.block_height + 1
        render_target.blit(
            self.cursor_sprite,
            (screen_x, screen_y)
        )

