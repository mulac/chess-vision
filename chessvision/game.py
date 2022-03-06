import os
import pickle
import itertools
import tempfile
import cv2
import chess.pgn

from .storage import Storage
from .label import label, find_corners, SIZE, MARGIN

LABELS =  [
    chess.Piece(piece_type, color) 
    for piece_type, color in itertools.product(chess.PIECE_TYPES, chess.COLORS)
]

id_to_label = {hash(l): l for l in LABELS}

class Game:
    def __init__(self, name, number, flipped=False, game_dir="games", skip_moves=2, board_size=SIZE, margin=MARGIN):
        self.__dict__.update(locals())
        self.pgn_path = os.path.abspath(os.path.join(
            self.game_dir, f"{self.name}.pgn"))
        self.pkl_path = os.path.abspath(os.path.join(
            self.game_dir, f"{self.name}_{self.number}.pkl"))

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
        return sum(1 for _ in self.images) - 2

    def __repr__(self):
        return (
            f'Game({self.name}, {self.number}, '
            f'flipped={self.flipped}, skip_moves={self.skip_moves}, '
            f'board_size={self.board_size}, margin={self.margin})'
        )

    def label(self):
        return label(
            self.pgn, 
            find_corners(self.images), 
            self.images,
            flipped=self.flipped,
            skip_moves=self.skip_moves,
            size=self.board_size,
            margin=self.margin
        )

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