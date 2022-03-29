import os
import torch
import cv2
import math
import time

from collections import deque

from . import label
from .record import Camera

id_to_label = {hash(lbl): lbl for lbl in label.PIECE_LABELS}


class LiveInference:
    def __init__(self, config, model, occupancy_model, device, camera,
        history=20, 
        motion_thresh=0
        ):
        if occupancy_model is None:
            self.occupancy_fn = self.occupancy_depth
        else:
            self.occupancy_fn = self.occupancy_nn
            occupancy_model.to(device)
            occupancy_model.eval()
        model.to(device)
        model.eval()

        self.begin = time.time()
        self.model = model
        self.config = config
        self.device = device
        self.camera = camera
        self.motion_thresh = motion_thresh

        self.corners = None
        self.prev = None
        self.history = [deque((None for _ in range(history)), history) for _ in range(64)]

    def memory(self):
        return [max(square, key=square.count) for square in self.history]

    def show_img(self, img, name="chess-vision", size=(960, 540)):
        cv2.imshow(name, cv2.resize(img, size))
        cv2.waitKey(1)
    
    def get_corners(self, img, _):
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
        preds = {}
        if len(occupied) > 0:
            squares =torch.stack(
                [self.config.infer_transform(label.get_square(i, board)) for i in occupied]
            ).to(self.device)
            preds = {occupied[i]: pred for i, pred in enumerate(self.model(squares).argmax(dim=1))}
        return preds

    def has_motion(self, current):
        if self.prev is None:
            self.prev = current
            return True

        def prepare(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.GaussianBlur(gray, (7, 7), 0)

        delta = cv2.absdiff(prepare(self.prev), prepare(current))
        _, thresh = cv2.threshold(delta, 20, 255, cv2.THRESH_BINARY) 
        self.prev = current
        self.show_img(delta, "delta", size=(500, 500))
        return thresh.mean() > self.motion_thresh

    def run_inference(self, img, depth):
        board = label.get_board(img, self.corners)
        if self.has_motion(board):
            return
        occupied = self.occupancy_fn(img, depth)
        preds = self.get_predictions(board, occupied)
        for i in range(64):
            self.history[i].append(preds.get(i))
        self.show_img(board, size=(500, 500))
        self.print_fen()

    def occupancy_nn(self, img, _):
        """ Uses a torch model to detect occupied squares 
        
        Returns: Dict[square_id: square_img]
        """
        squares = torch.stack(
            [self.config.infer_transform(square) for square in label.get_squares(img)]
        ).to(self.device)
        return [i for i, occupied in enumerate(self.model(squares).argmax(dim=1)) if occupied]

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
    def print_fen(self):
        t = int(time.time())
        self.count += 1
        if self.begin != t:
            print(self.count)
            self.count = 0
            self.begin = t
        print([id_to_label[i.item()].unicode_symbol() for i in self.memory() if i is not None])

    def start(self):
        try:
            self.camera.loop(self.get_corners)
            print(f"found corners... \n {self.corners}\n")
            self.camera.loop(self.run_inference)
        finally:
            cv2.destroyAllWindows()
            self.camera.close()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(os.path.join(args.dir, args.model, "model"), device)
    config = torch.load(os.path.join(args.dir, args.model, "config"), device)
    occupancy_model = (torch.load(os.path.join(args.dir, args.occupancy_model, "model"), device) if
        args.occupancy_model is not None else None)

    game = LiveInference(
        config,
        model,
        occupancy_model,
        device,
        Camera(depth=True),
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
    parser.add_argument('--dir', type=str, metavar='directory', default='models',
                        help='the directory the run can be found in')

    main(parser.parse_args())