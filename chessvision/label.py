""" Automatic labelling functions using the Game class """

import itertools
import numpy as np
import chess
import cv2

from collections import namedtuple

from .aruco import detect, DetectionError

SIZE = 800
MARGIN = 0
CUT = 5

Labeller = namedtuple('Labeller', ['labels', 'label_fn', 'names'])

PIECE_LABELS = [chess.Piece(piece_t, col) for piece_t, col in itertools.product(chess.PIECE_TYPES, chess.COLORS)]
OCCUPIED_LABELS = [True, False]
COLOR_LABELS = chess.COLORS
TYPE_LABELS = chess.PIECE_TYPES

_piece_by_id = {hash(lbl): lbl for lbl in PIECE_LABELS}
_piece_str_by_id = {hash(lbl): str(lbl) for lbl in PIECE_LABELS}

def from_id(id): return _piece_by_id[id]
def str_from_id(id): return _piece_str_by_id[id]


def label(game):
    corners = find_corners(game.images)
    images = skip(game.images, game.skip_moves)
    for move, img in zip(game.pgn.mainline(), images):
        yield label_move(move.board(), img['color'], move.move.to_square, corners, game.flipped, game.board_size, game.margin)

def label_color(game):
    for img, piece in label(game):
        yield img, piece.color

def label_type(game):
    for img, piece in label(game):
        yield img, piece.piece_type

def label_occupied(game, stream='color'):
    corners = find_corners(game.images)
    images = skip(game.images, game.skip_moves-1)
    start = next(images)
    for square in range(64):
        yield label_occupied_move(chess.Board(), start[stream], square, corners, game.flipped, game.board_size, game.margin)
    for move, img in zip(game.pgn.mainline(), images):
        yield label_occupied_move(move.board(), img[stream], move.move.to_square, corners, game.flipped, game.board_size, game.margin)
        yield label_occupied_move(move.board(), img[stream], move.move.from_square, corners, game.flipped, game.board_size, game.margin)
 

# ===========================================================
# The following are helpers for the above labelling functions
# ===========================================================

def skip(iterator, n):
    for _ in range(n):
        next(iterator)
    return iterator


def label_occupied_move(pgn_board, img, square, corners=None, flipped=False, size=SIZE, margin=MARGIN):
    piece_img, piece = label_move(pgn_board, img, square, corners, flipped, size, margin)
    return piece_img, piece is not None


def find_corners(images):
    for img in images:
        try:
            return get_corners(img["color"])
        except DetectionError:
            continue

    raise DetectionError("failed to find markers from any move")


def get_corners(image):
    return np.array([c[1][0][0] for c in detect(image)])


def label_move(pgn_board, img, square, corners=None, flipped=False, size=SIZE, margin=MARGIN):
    board_img = get_board(img, corners, size, margin) if corners is not None else img
    # BUG should we be flipping on axis 1?
    piece_img = get_square(square, np.flip(board_img) if flipped else board_img, size, margin)
    label = pgn_board.piece_at(square)
    return piece_img, label

def get_occupied_squares(depth_img, corners, size=SIZE, margin=MARGIN, cut=CUT):
    def occupied(img):
        img = img[cut:-cut, cut:-cut].flatten()
        return np.sum(img * (img < 255))
    depth_squares = get_squares(get_board(depth_img, corners), size=size, margin=margin)
    return [i for i, square in enumerate(depth_squares) if occupied(square)]


def get_board(img, corners, size=SIZE, margin=MARGIN):
    dest = np.float32([
        [margin, margin],
        [size+margin, margin],
        [margin, size+margin],
        [size+margin, size+margin]
    ])
    transform = cv2.getPerspectiveTransform(corners, dest)
    return cv2.warpPerspective(img, transform, (size+2*margin, size+2*margin))


def get_squares(img, size=SIZE, margin=MARGIN):
    return (get_square(i, img, size, margin) for i in range(64))


def get_square(square, img, size=SIZE, margin=MARGIN):
    square_size = size // 8
    i = 7 - square % 8
    j = 7 - square // 8
    top_x = i * square_size
    bot_x = (i+1)*square_size + 2*margin
    top_y = j*square_size
    bot_y = (j+1)*square_size + 2*margin
    return img[top_y:bot_y, top_x:bot_x]