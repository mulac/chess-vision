import os
import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy

import label
import record


class LiveInference:
    def __init__(self, model, config, device, camera):
        model.to(device)
        model.eval()

        self.model = model
        self.config = config
        self.device = device
        self.camera = camera
        self.corners = None

    def show_img(self, img, size=(960, 540)):
        cv2.imshow('chess-vision', cv2.resize(img, size))
        cv2.waitKey(1)
    
    def get_corners(self, img, _):
        self.show_img(img)
        try: 
            self.corners = label._get_corners(img)
        except label.aruco.DetectionError:
            return
        return record.Camera.cancel_signal

    def get_piece_predictions(self, img, depth):
        self.depth(depth)
        return
        board = label.get_board(img, self.corners)
        b = label.get_board(depth, self.corners)
        self.show_img(board, size=(500, 500))
        squares = [self.config.infer_transform(square) for square in label.get_squares(board)]
        squares = torch.stack(squares).to(self.device)
        pred = self.model(squares)
        self.print_fen(pred.argmax(dim=1))

    def depth(self, depth):
        board = label.get_board(depth, self.corners)
        board = cv2.applyColorMap(cv2.convertScaleAbs(board, alpha=0.03), cv2.COLORMAP_JET)
        self.show_img(board, size=(500, 500))
        squares = label.get_squares(board)

        print(scipy.spatial.KDTree([np.sum(square) for square in squares]))

    def print_fen(self, pieces):
        labels = [str(label.to_label(i)) for i in pieces]
        # bottom right, bottom left, top right, top left
        print(labels[27], labels[28], labels[35], labels[36])

    def start(self):
        try:
            self.camera.loop(self.get_corners)
            print(f"found corners... \n {self.corners}\n")
            self.camera.loop(self.get_piece_predictions)
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
        record.Camera(depth=True)
    )

    game.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Will load an archived model and begin inference from the camera stream.')
    parser.add_argument('model', type=str,
                        help='the id of a run to use for inference')
    parser.add_argument('--dir', type=str, metavar='directory', default='models',
                        help='the directory the run can be found in')

    main(parser.parse_args())