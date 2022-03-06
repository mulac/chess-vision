import os
import torch
import cv2
import math
import time

from collections import deque

from . import label
from .game import id_to_label
from .record import Camera


class LiveInference:
    def __init__(self, model, config, device, camera, history=20):
        model.to(device)
        model.eval()

        self.begin = time.time()
        self.model = model
        self.config = config
        self.device = device
        self.camera = camera

        self.corners = None
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
        preds = {}
        if len(occupied) > 0:
            squares =torch.stack(
                [self.config.infer_transform(label.get_square(i, board)) for i in occupied]
            ).to(self.device)
            preds = {occupied[i]: pred for i, pred in enumerate(self.model(squares).argmax(dim=1))}
        return preds

    def run_inference(self, img, depth):
        board = label.get_board(img, self.corners)
        occupied = self.depth(depth)
        preds = self.get_predictions(board, occupied)
        for i in range(64):
            self.history[i].append(preds.get(i))
        self.show_img(board, size=(500, 500))
        self.print_fen()

    def depth(self, depth):
        board = label.get_board(depth, self.corners)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(board, alpha=0.03), cv2.COLORMAP_JET)
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
    model_path = os.path.join(args.dir, args.model, "model")
    config_path = os.path.join(args.dir, args.model, "config")

    game = LiveInference(
        torch.load(model_path), 
        torch.load(config_path), 
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        Camera(depth=True)
    )

    game.start()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Will load an archived model and begin inference from the camera stream.')
    parser.add_argument('model', type=str,
                        help='the id of a run to use for inference')
    parser.add_argument('--dir', type=str, metavar='directory', default='models',
                        help='the directory the run can be found in')

    main(parser.parse_args())