import os
import pickle
import chess.pgn
import numpy as np
import cv2
import aruco


_size = 800
_margin = 0


class Game:
    def __init__(self, name, number, game_dir="games"):
        self.name = name
        self.number = number
        self.game_dir = game_dir

        self.pgn_path = os.path.abspath(os.path.join(
            self.games_dir, f"{self.name}.pgn"))
        self.pkl_path = os.path.abspath(os.path.join(
            self.games_dir, f"{self.name}_{self.number}.pkl"))

    @property
    def pgn(self):
        with open(self.pgn_path) as pgn:
            for i in range(self.number):
                chess.pgn.skip_game(pgn)
            return chess.pgn.read_game(pgn)

    @property
    def images(self):
        with open(self.pkl_path, "rb") as pkl:
            return pickle.load(pkl)


def label(game):
    return _label(game.pgn, get_corners(game.images), game.images)


def _label(pgn_game, corners, images, size=_size, margin=_margin):
    for move, img in zip(pgn_game.mainline(), images):
        yield label_move(move, img["color"], corners, size=size, margin=margin)


def get_corners(images):
    for img in images:
        try:
            return np.array([
                c[1][0][0] for c in aruco.detect(img["color"])
            ])
        except aruco.DetectionError:
            continue

    raise aruco.DetectionError("failed to find markers from any move")


def label_move(move, img, corners, size=_size, margin=_margin):
    board_img = get_board(img, corners, size, margin)
    piece_img = get_square(move.move.to_square, board_img, size, margin)
    label = move.board().piece_at(move.move.to_square)
    return piece_img, label


def get_board(img, corners, size=_size, margin=_margin):
    dest = np.float32([
        [margin, margin],
        [size+margin, margin],
        [margin, size+margin],
        [size+margin, size+margin]
    ])

    transform = cv2.getPerspectiveTransform(corners, dest)
    return cv2.warpPerspective(img, transform, (size+2*margin, size+2*margin))


def get_square(square, img, size=_size, margin=_margin):
    square_size = size // 8
    i = 7 - square % 8
    j = 7 - square // 8
    top_x = i * square_size
    bot_x = (i+1)*square_size + 2*margin
    top_y = j*square_size
    bot_y = (j+1)*square_size + 2*margin
    return img[top_y:bot_y, top_x:bot_x]


def get_squares(img, size=_size, margin=_margin):
    square_size = size // 8
    squares = []
    for i in reversed(range(8)):
        top_y = i*square_size
        bot_y = (i+1)*square_size + 2*margin
        for j in reversed(range(8)):
            top_x = j*square_size
            bot_x = (j+1)*square_size + 2*margin
            squares.append(img[top_y:bot_y, top_x:bot_x])
    return squares
