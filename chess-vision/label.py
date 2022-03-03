import os
import pickle
import itertools
import tempfile
import numpy as np
import cv2
import chess.pgn
import aruco

from storage import Storage

_size = 800
_margin = 0

LABELS =  [
    chess.Piece(piece_type, color) 
    for piece_type, color in itertools.product(chess.PIECE_TYPES, chess.COLORS)
]

def to_label(i):
    for label in LABELS:
        if hash(label) == i:
            return label


class Game:
    def __init__(self, name, number, flipped=False, game_dir="games", skip_moves=2, board_size=_size, margin=_margin):
        self.__dict__.update(locals())
        self.pgn_path = os.path.abspath(os.path.join(
            self.game_dir, f"{self.name}.pgn"))
        self.pkl_path = os.path.abspath(os.path.join(
            self.game_dir, f"{self.name}_{self.number}.pkl"))
        self.length = sum(1 for _ in self.images) - 2

    @property
    def pgn(self):
        with open(Storage(self.pgn_path)) as pgn:
            for i in range(self.number):
                chess.pgn.skip_game(pgn)
            return chess.pgn.read_game(pgn)

    @property
    def images(self):
        with open(Storage(self.pkl_path), "rb") as pkl:
            while True:
                try:
                    yield pickle.load(pkl)
                except EOFError:
                    break

    def __len__(self):
        return self.length

    def __repr__(self):
        return (
            f'Game({self.name}, {self.number}, '
            f'flipped={self.flipped}, skip_moves={self.skip_moves}, '
            f'board_size={self.board_size}, margin={self.margin})'
        )

    def label(self):
        return label(
            self.pgn, 
            get_corners(self.images), 
            self.images,
            flipped=self.flipped,
            skip_moves=self.skip_moves,
            size=self.board_size,
            margin=self.margin
        )


def label(pgn_game, corners, images, flipped=False, skip_moves=2, size=_size, margin=_margin):
    skip(images, skip_moves)
    for move, img in zip(pgn_game.mainline(), images):
        img = img["color"] if not flipped else np.flip(img["color"])
        yield label_move(move.board(), get_board(img, corners, size, margin), move.move.to_square, size, margin)


def skip(iterator, i):
    for _ in range(i):
        next(iterator)


def get_corners(images):
    for img in images:
        try:
            return _get_corners(img["color"])
        except aruco.DetectionError:
            continue

    raise aruco.DetectionError("failed to find markers from any move")


def _get_corners(image):
    return np.array([c[1][0][0] for c in aruco.detect(image)])


def label_move(pgn_board, board_img, square, size=_size, margin=_margin):
    piece_img = get_square(square, board_img, size, margin)
    label = pgn_board.piece_at(square)
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


def save_games(games, root_dir=None):
    root_dir, label_dirs = create_dirs(root_dir)
    for game in games:
        for img, lbl in game.label():
            _, path = tempfile.mkstemp(suffix=".jpg", dir=label_dirs[lbl])
            cv2.imwrite(path, img)
    return root_dir


def create_dirs(root_dir=None):
    if root_dir is None:
        root_dir = tempfile.mkdtemp(prefix="chess-vision-")
    label_dirs = {lbl: os.path.join(root_dir, str(hash(lbl))) for lbl in LABELS}
    for label in label_dirs:
        os.mkdir(label_dirs[label])
    return root_dir, label_dirs
