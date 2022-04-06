""" An abstraction for interfacing with one unit of recorded data in a pickle file """

import os
import pickle
import tempfile
import cv2
import chess
import chess.pgn

from .storage import Storage
from .label import LabelOptions


class Game:
    """ Game is a wrapper for pulling images and pgn data from a pickle file """
    def __init__(self, name, number, options=LabelOptions()):
        self.name, self.number, self.options = name, number, options
        self.pgn_file = f"{self.name}.pgn"
        self.pkl_file = f"{self.name}_{self.number}.pkl"

    def __len__(self):
        return sum(1 for _ in self.images) - self.skip_moves

    def __repr__(self):
        return f'Game({self.name}, {self.number}, {self.options})'

    @property
    def pgn(self):
        with open(Storage(self.pgn_file)) as pgn:
            for i in range(self.number):
                chess.pgn.skip_game(pgn)
            return chess.pgn.read_game(pgn)

    @property
    def images(self):
        with open(Storage(self.pkl_file), "rb") as pkl:
            while True:
                try:
                    yield pickle.load(pkl)
                except EOFError:
                    break


def save_games(games, label_fn, labels, root_dir=None):
    """ Save a dataset from a set of game onto disk.
    Images are grouped by label with the label being the parent directory name.
    """
    root_dir, label_dirs = _create_dirs(labels, root_dir)
    for game in games:
        for img, lbl in label_fn(game):
            _, path = tempfile.mkstemp(suffix=".jpg", dir=label_dirs[lbl])
            cv2.imwrite(path, img)
    return root_dir


def _create_dirs(labels, root_dir=None):
    if root_dir is None:
        root_dir = tempfile.mkdtemp(prefix="chess-vision-")
    label_dirs = {lbl: os.path.join(root_dir, str(hash(lbl))) for lbl in labels}
    for label in label_dirs:
        os.mkdir(label_dirs[label])
    return root_dir, label_dirs