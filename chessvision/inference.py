import os
import time
import torch
import cv2
import chess
import chess.pgn
import numpy as np

from collections import deque
from cairosvg import svg2png

from . import label
from .camera import Camera as Camera, RealsenseCamera

class BoardState(chess.Board):
    """ BoardState is what we update to keep track of a full valid game """
    def update(self, board):
        if not isinstance(board, VisionState):
            board = VisionState(board)
        if not board.is_valid: return False
        for move in self.legal_moves:
            self.push(move)
            if self == board: return True
            self.pop()
        return False

class VisionState(chess.Board):
    """ VisionState is a potentially invalid board state """
    def __init__(self, board):
        super().__init__()
        self.set_piece_map({i: label.from_id(piece.item()) for i, piece in enumerate(board) if piece is not None})

class LiveInference:
    def __init__(self, config, model, occupancy_model, occupancy_config, device, camera,
        history=20, 
        motion_thresh=20
        ):
        if occupancy_model is None:
            self.occupancy_fn = self.occupancy_depth
        else:
            self.occupancy_fn = self.occupancy_nn
            self.occupancy_model = occupancy_model
            self.occupancy_config = occupancy_config
            self.occupancy_model.to(device)
            self.occupancy_model.eval()
        model.to(device)
        model.eval()

        self.begin = time.time()
        self.model = model
        self.config = config
        self.device = device
        self.camera = camera
        self.motion_thresh = motion_thresh

        self.board = BoardState()

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

    def get_predictions(self, board, occupied):
        """ Sends in all the occupied squares into the model as a batch 
        
        Returns: Dict[square_id: prediciton_id]
        """
        if len(occupied) == 0: return {}
        squares = torch.stack(
            [self.config.infer_transform(label.get_square(i, board)) for i in occupied]
        ).to(self.device)
        return {occupied[i]: pred for i, pred in enumerate(self.model(squares).argmax(dim=1))}

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
        if img is None: time.sleep(100)
        board = label.get_board(img, self.corners)
        if self.has_motion(board):
            return
        occupied = self.occupancy_fn(board)
        preds = self.get_predictions(board, occupied)
        for i in range(64):
            self.history[i].append(preds.get(i))
        self.show_img(board, size=(500, 500))
        self.print_fen()
        if self.board.is_game_over():
            print(self.board, self.board.is_game_over())
            return Camera.cancel_signal

    def occupancy_nn(self, img):
        """ Uses a torch model to detect occupied squares 
        
        Returns: Dict[square_id: square_img]
        """
        squares = torch.stack(
            [self.occupancy_config.infer_transform(square) for square in label.get_squares(img)]
        ).to(self.device)
        return [i for i, occupied in enumerate(self.occupancy_model(squares).argmax(dim=1)) if occupied.item() == 1]

    def occupancy_depth(self, _, depth):
        board = label.get_board(depth, self.corners)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(board, alpha=0.03), cv2.COLORMAP_JET)
        square_offset = label.SIZE // 16
        for i in range(1, 9):
            for j in range(1, 9):
                x = label.SIZE * i // 8 - square_offset
                y = label.SIZE * j // 8 - square_offset
                sp = (x - label.CUT, y - label.CUT)
                ep = (x + label.CUT, y + label.CUT)
                depth_colormap = cv2.rectangle(depth_colormap, sp, ep, (0, 0, 255), 2)
        self.show_img(depth_colormap, name="depth", size=(500, 500))
        return label.get_occupied_squares(depth, self.corners)

    count = 0
    def print_frame_rate(self):
        self.count += 1
        if t := int(time.time()) != self.begin:
            print(self.count)
            self.count = 0
            self.begin = t

    def print_fen(self):
        if self.board != (vision := self.memory()):
            svg_img = np.frombuffer(svg2png(vision._repr_svg_()), dtype=np.uint8)
            self.show_img(cv2.imdecode(svg_img, cv2.IMREAD_COLOR), "vision", (400, 400))
            print(f"\n\nVisionState:\n{vision}")
            print("\nBoardState Changed:", self.board.update(vision))
            # print("changed:", [label.from_id[i.item()].unicode_symbol() for i in board if i is not None])

    def start(self):
        try:
            self.camera.loop(self.get_corners)
            print(f"found corners... \n {self.corners}\n")
            self.camera.loop(self.run_inference)
            
            chess.pgn.Game.from_board(self.board).accept(
                chess.pgn.FileExporter(open("game.pgn", "w", encoding="utf-8"))
            )
        finally:
            cv2.destroyAllWindows()
            self.camera.close()


def main(args):
    if args.video:
        if os.path.exists(args.video): camera = Camera(args.video)
        else: exit(f"can't read video stream: {args.video} does not exist")
    else:
        camera = RealsenseCamera()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(os.path.join(args.dir, args.model, "model"), device)
    config = torch.load(os.path.join(args.dir, args.model, "config"), device)
    occupancy_model = (torch.load(os.path.join(args.dir, args.occupancy_model, "model"), device) if
        args.occupancy_model is not None else None)
    occupancy_config = (torch.load(os.path.join(args.dir, args.occupancy_model, "config"), device) if
        args.occupancy_model is not None else None)

    game = LiveInference(
        config,
        model,
        occupancy_model,
        occupancy_config,
        device,
        camera,
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
    parser.add_argument('--video', type=str, metavar='video',
                        help='the path of a video stream to use')
    parser.add_argument('--dir', type=str, metavar='directory', default='runs',
                        help='the directory the run can be found in')

    main(parser.parse_args())