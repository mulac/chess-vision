import os
import sys
import time
import torch
import cv2
import chess
import chess.pgn
import numpy as np

from select import select
from collections import deque
from cairosvg import svg2png

from . import label
from .camera import Camera as Camera, RealsenseCamera


class BoardState(chess.Board):
    """ BoardState is what we update to keep track of a full valid game """
    def update(self, board):
        if not isinstance(board, VisionState):
            board = VisionState(board)
        for move in self.legal_moves:
            self.push(move)
            if self == board: 
                return True
            self.pop()
        return False


class VisionState(chess.BaseBoard):
    """ VisionState is a potentially invalid board state """
    def __init__(self, board):
        super().__init__()
        self._set_piece_map({i: label.PIECE_LABELS[piece] for i, piece in enumerate(board) if piece is not None})


class LiveInference:
    def __init__(self, config, model, occupancy_model, occupancy_config, device, camera,
        color_model=None, 
        color_config=None,
        history=20, 
        motion_thresh=20
        ):
        self.occupancy_model = occupancy_model
        self.occupancy_config = occupancy_config
        self.occupancy_model.to(device)
        self.occupancy_model.eval()

        self.model = model
        self.config = config
        self.model.to(device)
        self.model.eval()
        
        self.color_model = color_model
        self.color_config = color_config
        if color_model is not None:
            self.color_model.to(device)
            self.color_model.eval()

        self.begin = time.time()
        
        self.device = device
        self.camera = camera
        self.motion_thresh = motion_thresh

        self.board = BoardState()
        self.board_svg = np.frombuffer(svg2png(self.board._repr_svg_()), dtype=np.uint8)

        self.corners = None
        self.prev_img = None
        self.history = [deque((None for _ in range(history)), history) for _ in range(64)]

    def memory(self):
        return VisionState([max(square, key=square.count) for square in self.history])

    def show_img(self, img, name="chess-vision", size=(960, 540)):
        cv2.imshow(name, cv2.resize(img, size))
        cv2.waitKey(1)
    
    def get_corners(self, img, _):
        if img is None: 
            return
        self.show_img(img)
        try: 
            self.corners = label.get_corners(img)
        except label.DetectionError: 
            return
        return Camera.cancel_signal

    def get_predictions(self, board):
        """ Sends in all the occupied squares into the model as a batch 
        
        Returns: Dict[square_id: prediciton_id]
        """
        squares = list(label.get_squares(board))
        
        occupied = [i for i, occupied in enumerate(self.occupancy_model(torch.stack(
            [self.occupancy_config.infer_transform(square) for square in squares]
        ).to(self.device)).argmax(dim=1)) if occupied.item() == 0]
        if len(occupied) == 0: 
            return {}
        
        pieces = self.model(torch.stack(
            [self.config.infer_transform(squares[i]) for i in occupied]
        ).to(self.device)).argmax(dim=1)

        if self.color_model is not None:
            colors = self.color_model(torch.stack(
                [self.color_config.infer_transform(squares[i]) for i in occupied]
            ).to(self.device)).argmax(dim=1)
            pieces = pieces + 6 * (1 - colors)
        
        return {occupied[i]: pred.item() for i, pred in enumerate(pieces)}

    def has_motion(self, current):
        if self.prev_img is None:
            self.prev_img = current
            return True

        def prepare(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.GaussianBlur(gray, (7, 7), 0)

        delta = cv2.absdiff(prepare(self.prev_img), prepare(current))
        _, thresh = cv2.threshold(delta, self.motion_thresh, 255, cv2.THRESH_BINARY) 
        self.prev_img = current
        self.show_img(delta, "delta", size=(500, 500))
        return thresh.sum() > 0

    def run_inference(self, img, _):
        if img is None:
            # FOR DEBUGGING -> REMOVE
            time.sleep(100)
        board = label.get_board(img, self.corners)
        # if self.has_motion(board):
        #     return
        preds = self.get_predictions(board)
        for i in range(64):
            self.history[i].append(preds.get(i))
        self.show_img(board, size=(500, 500))
        self.display()
        if self.board.is_game_over():
            print(self.board, self.board.is_game_over())
            return Camera.cancel_signal

    def display(self):
        if select([sys.stdin], [], [], 0)[0] != []:
            sys.stdin.readline()
            print("snapshot saved")
        if self.board != (vision := self.memory()):
            svg_img = np.frombuffer(svg2png(vision._repr_svg_()), dtype=np.uint8)
            self.show_img(cv2.imdecode(svg_img, cv2.IMREAD_COLOR), "vision", (400, 400))
            # print(f"\n\nVisionState:\n{vision}")
            if self.board.update(vision):
                self.board_svg = np.frombuffer(svg2png(self.board._repr_svg_()), dtype=np.uint8)
            self.show_img(cv2.imdecode(self.board_svg, cv2.IMREAD_COLOR), "board", (400, 400))

    def start(self):
        try:
            self.camera.loop(self.get_corners)
            print(f"found corners... \n {self.corners}\n")
            print("Press [ENTER] to record a snapshot")
            self.camera.loop(self.run_inference)
        finally:
            chess.pgn.Game.from_board(self.board).accept(
                chess.pgn.FileExporter(open("game.pgn", "w", encoding="utf-8"))
            )
            cv2.destroyAllWindows()
            self.camera.close()


def main(args):
    if args.video:
        camera = Camera(args.video)
        # if os.path.exists(args.video): camera = Camera(args.video)
        # else: exit(f"can't read video stream: {args.video} does not exist")
    else:
        camera = RealsenseCamera()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(os.path.join(args.dir, args.model, "model"), device)
    config = torch.load(os.path.join(args.dir, args.model, "config"), device)
    occupancy_model = (torch.load(os.path.join(args.dir, args.occupancy_model, "model"), device) if
        args.occupancy_model is not None else None)
    occupancy_config = (torch.load(os.path.join(args.dir, args.occupancy_model, "config"), device) if
        args.occupancy_model is not None else None)
    color_model = (torch.load(os.path.join(args.dir, args.color_model, "model"), device) if
        args.color_model is not None else None)
    color_config = (torch.load(os.path.join(args.dir, args.color_model, "config"), device) if
        args.color_model is not None else None)

    game = LiveInference(
        config,
        model,
        occupancy_model,
        occupancy_config,
        device,
        camera,
        color_model=color_model,
        color_config=color_config,
        history=1
    )

    game.start()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Will load an archived model and begin inference from the camera stream.')
    parser.add_argument('model', type=str,
                        help='the id of a run to use for piece inference')
    parser.add_argument('--occupancy-model', type=str, metavar='occupancy_model',
                        help='the id of a run to use for occupancy inference')
    parser.add_argument('--color-model', type=str, metavar='color_model',
                        help='the id of a run to use for color inference')
    parser.add_argument('--video', type=str, metavar='video',
                        help='the path of a video stream to use')
    parser.add_argument('--dir', type=str, metavar='directory', default='runs',
                        help='the directory the run can be found in')

    main(parser.parse_args())