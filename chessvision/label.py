""" Automatic labelling functions using the Game class """

import numpy as np
import chess
import cv2

from itertools import product, chain
from dataclasses import dataclass

from .util import skip
from .aruco import detect, DetectionError


class Labeller:
    def __init__(self, label_fn, classes, names):
        assert len(classes) == len(names)
        self.label_fn, self.classes, self.names = label_fn, classes, names
        
    def __call__(self, x):
        return self.label_fn(x)

    def __repr__(self):
        return f'Labeller({zip(self.classes, self.names)})'

    def __iter__(self):
        return iter(enumerate(self.classes))


@dataclass
class LabelOptions():
    """ LabelOptions change the behavior of the auto-labelling functions """
    board_size: int = 800
    margin: int = 0
    cut: int = 5
    skip_moves: int = 2
    flipped: bool = False



# Different sets of label classes
# ================================
PIECE_LABELS    = [chess.Piece(piece_t, color) for piece_t, color in product(chess.PIECE_TYPES, chess.COLORS)]
OCCUPIED_LABELS = [OCCUPIED, EMPTY] = [True, False]
COLOR_LABELS    = [WHITE, BLACK] = [True, False]
TYPE_LABELS     = [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING] = range(1, 7)
TYPE_LABELS_P   = [EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING]
ALL_LABELS      = PIECE_LABELS + [EMPTY]


# Different label functions for each label class above
# ====================================================
def label(game, stream='color'):
    corners = find_corners(game.images)
    images = skip(game.images, game.options.skip_moves-1)
    start_img = next(images)
    for square in range(64):
        yield label_square(chess.Board(), start_img[stream], square, corners, game.options)
    for move, img in zip(game.pgn.mainline(), images):
        yield label_square(move.board(), img[stream], move.move.to_square, corners, game.options)
        yield label_square(move.board(), img[stream], move.move.from_square, corners, game.options)

def label_pieces(game, stream='color'):
    corners = find_corners(game.images)
    images = skip(game.images, game.options.skip_moves-1)
    start_img = next(images)
    for square in chain(range(16), range(48, 64)):
        yield label_square(chess.Board(), start_img[stream], square, corners, game.options)
    for move, img in zip(game.pgn.mainline(), images):
        yield label_square(move.board(), img[stream], move.move.to_square, corners, game.options)

def label_with_board(game, stream='color'):
    corners = find_corners(game.images)
    images = skip(game.images, game.options.skip_moves-1)
    img = next(images)
    for square in chain(range(16), range(48, 64)):
        square, lbl = label_square(chess.Board(), img[stream], square, corners, game.options)
        yield (img[stream], square), lbl.color
    for move, img in zip(game.pgn.mainline(), images):
        square, lbl = label_square(move.board(), img[stream], move.move.to_square, corners, game.options)
        yield (img[stream], square), lbl.color

def label_color(game):
    for img, piece in label_pieces(game):
        yield img, piece.color

def label_type(game):
    for img, piece in label_pieces(game):
        yield img, piece.piece_type

def label_type_p(game):
    for img, piece in label(game):
        yield img, EMPTY if piece is EMPTY else piece.piece_type

def label_occupied(game, stream='color'):
    for img, piece in label(game, stream):
        yield img, EMPTY if piece is EMPTY else OCCUPIED
 

# ===========================================================
# The following are helpers for the above labelling functions
# ===========================================================

def find_corners(images):
    for img in images:
        try:
            return get_corners(img["color"])
        except DetectionError:
            continue

    raise DetectionError("failed to find markers from any move")


def get_corners(image):
    return np.array([c[1][0][0] for c in detect(image)])


def label_occupied_move(pgn_board, img, square, corners=None, opts=LabelOptions()):
    piece_img, piece = label_square(pgn_board, img, square, corners, opts)
    return piece_img, piece is not EMPTY


def label_square(pgn_board, img, square, corners=None, opts=LabelOptions()):
    board_img = get_board(img, corners, opts) if corners is not None else img
    piece_img = get_square(square, board_img, opts)
    label = pgn_board.piece_at(square)
    return piece_img, label if label is not None else EMPTY


def get_occupied_squares(depth_img, corners, opts=LabelOptions()):
    def occupied(img):
        img = img[opts.cut:-opts.cut, opts.cut:-opts.cut].flatten()
        return np.sum(img * (img < 255))
    depth_squares = get_squares(get_board(depth_img, corners, opts), opts)
    return [i for i, square in enumerate(depth_squares) if occupied(square)]


def get_board(img, corners, opts=LabelOptions()):
    margin, size, flipped = opts.margin, opts.board_size, opts.flipped
    dest = np.float32([
        [margin, margin],
        [size+margin, margin],
        [margin, size+margin],
        [size+margin, size+margin]
    ])
    if flipped: dest = dest[[1, 0, 3, 2]]
    transform = cv2.getPerspectiveTransform(corners, dest)
    return cv2.warpPerspective(img, transform, (size+2*margin, size+2*margin))


def get_squares(img, opts=LabelOptions()):
    return (get_square(i, img, opts) for i in range(64))


def get_square(square, img, opts=LabelOptions()):
    square_size = opts.board_size // 8
    i = 7 - square % 8
    j = 7 - square // 8
    top_x = i * square_size
    bot_x = (i+1)*square_size + 2*opts.margin
    top_y = j*square_size
    bot_y = (j+1)*square_size + 2*opts.margin
    return img[top_y:bot_y, top_x:bot_x]