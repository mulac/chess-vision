""" An abstraction for interfacing with one unit of recorded data in a pickle file """

import os
import pickle
import logging
from random import shuffle
import tempfile
import cv2
import chess
import chess.pgn

from torchvision import datasets

from .storage import Storage
from .label import *


class ChessFolder(datasets.ImageFolder):
    def find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        return classes, {i: int(i) for i in classes}


class Game:
    """ Game is a wrapper for pulling images and pgn data from a pickle file.
    PGN games can be downloaded from https://www.pgnmentor.com/files.html 
    """
    def __init__(self, name, number, options=LabelOptions()):
        self.name, self.number, self.options = name, number, options
        self.pgn_file = f'{name}.pgn'
        self.pkl_file = f'{name}_{number}.pkl'

    @classmethod
    def from_file(cls, fname, options=LabelOptions()):
        game = cls(*fname.split('_')[:2], options=options)
        game.pkl_file = fname
        return game

    def __len__(self):
        return sum(1 for _ in self.images) - self.options.skip_moves

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


def save_games(games, labeller, root_dir=None, distribution=None):
    """ Save a dataset from a set of games onto disk as a ChessFolder.
    Images are grouped by label with the label being the parent directory name.

    Args:
        games: Iterable[Game]
        labeller: Labeller
        root_dir: path to save games (default a temp directory)
        distribution: Dict[label: percent] percent=0.5 will half images for that label

    Returns:
        root_dir: path to saved games
    """
    root_dir, label_dirs = _create_dirs(labeller, root_dir)
    logging.info(f'saving games to {root_dir}...')
    for game in games:
        for img, lbl in labeller(game):
            fd, path = tempfile.mkstemp(suffix=".jpg", dir=label_dirs[lbl])
            cv2.imwrite(path, img)
            os.close(fd)
    if distribution is not None: rebalance(label_dirs, distribution)
    return root_dir


def rebalance(label_dirs, distribution):
    """ Rebalance an existing saved ChessFolder by a distribution.
    
    Args:
        label_dirs: Dict[label: sub_folder_path]
        distribution: Dict[label: percent] 0.8 will remove 20% of that labels images
    """
    for lbl, path in label_dirs:
        if lbl not in distribution or not (0 <= distribution[lbl] <= 1): 
            logging.warn(f'rebalance: could not valid value of {lbl} in {distribution}\
                            leaving {path} as is.')
            continue
        file_list = [f for f in os.listdir(path) if os.path.isfile(f)]
        for fname in shuffle(file_list)[int(len(file_list) * distribution[lbl]):]:
            os.remove(fname)



def _create_dirs(labeller: Labeller, root_dir=None):
    """ Creates the directory structure for a ChessFolder 

    Args:
        labeller: Labeller
        root_dir: path to save games (default a temp directory)

    Returns:
        root_dir: path to root directory
        label_dirs: Dict[label: sub_folder_path]
    """
    if root_dir is None:
        root_dir = tempfile.mkdtemp(prefix="chess-vision-")
    label_dirs = {lbl: os.path.join(root_dir, str(idx)) for idx, lbl in labeller}
    for label in label_dirs:
        os.mkdir(label_dirs[label])
    return root_dir, label_dirs