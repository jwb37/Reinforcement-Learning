import pygame

from . import BoardState

class PygameRenderer():
    def __init__(self, game, bgcolour=(180,180,255)):
        self.game = game
        self.board = game.board
        self.bgcolour = bgcolour

    def resize(self, display_width, display_height):
        grid_width = self.board.width - 4
        grid_height = self.board.height - 2

        self.board_surface = pygame.Surface( (display_width, display_height) )
        self.block_width = int(display_width/grid_width)
        self.block_height = int(display_height/grid_height)

        # TODO: Should probably set font size based on display_width and display_height
        #self.font = pygame.font.Font('EBGaramond08-Regular.ttf', 12)

        self.render_sprites()


    def render_sprites(self):
        w, h = self.block_width, self.block_height

        self.block_sprite = pygame.Surface( (w, h) )
        self.block_sprite.fill( (250, 250, 50) )

        self.piece_block_sprite = pygame.Surface( (w, h) )
        self.piece_block_sprite.fill( (100, 60, 0) )

        self.clearing_block_sprite = pygame.Surface( (w, h) )
        self.clearing_block_sprite.fill( (255, 0, 0) )

        # Draw black borders around all sprites
        for sprite in (self.block_sprite, self.piece_block_sprite, self.clearing_block_sprite):
            pygame.draw.aalines(
                sprite,
                (0,0,0),
                True,
                [ (0,0), (w-1,0), (w-1,h-1), (0,h-1) ]
            )

    # Renders just the board area
    def render_board(self, render_target):
        self.board_surface.fill( self.bgcolour )

        w = self.board.width - 4
        h = self.board.height - 2

        clearing_lines = self.board.state in (BoardState.Clearing, BoardState.Falling)

        # Draw board
        for y in range(h):

            if clearing_lines and y+2 in self.board.lines_to_clear:
                sprite = self.clearing_block_sprite
            else:
                sprite = self.block_sprite

            for x in range(w):
                if self.board.blocks[ x+2, y+2 ]:
                    self.board_surface.blit(
                        sprite,
                        (x * self.block_width, (h-y-1) * self.block_height)
                    )

        # Draw piece
        if not clearing_lines:
            for x in range(4):
                for y in range(4):
                    if self.board.piece.mask[ x, y ]:
                        self.board_surface.blit(
                            self.piece_block_sprite,
                            ((x + self.board.piece.x - 4) * self.block_width, (h - y - self.board.piece.y + 3) * self.block_height)
                        )

        render_target.blit(
            self.board_surface,
            (0, 0)
        )

    # This should render the whole game screen (possibly including scoring area to the side?
    def render(self, render_target):
        self.render_board(render_target)
        # Print score here
