import os
import torch
import cv2

from . import label
from .game import id_to_label
from .record import Camera


class LiveInference:
    def __init__(self, model, config, device, camera):
        model.to(device)
        model.eval()

        self.model = model
        self.config = config
        self.device = device
        self.camera = camera
        self.corners = None

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

    def get_piece_predictions(self, img, depth):
        board = label.get_board(img, self.corners)
        self.show_img(board, size=(500, 500))
        squares_idx = self.depth(depth)
        squares = [self.config.infer_transform(label.get_square(i, board)) for i in squares_idx]
        stacked_squares = torch.stack(squares).to(self.device)
        pred = self.model(stacked_squares)
        self.print_fen(pred.argmax(dim=1))

    def depth(self, depth):
        board = label.get_board(depth, self.corners)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(board, alpha=0.03), cv2.COLORMAP_JET)
        self.show_img(depth_colormap, name="depth", size=(500, 500))
        return label.get_occupied_squares(depth, self.corners)

    def print_fen(self, pieces):
        print([id_to_label[i.item()].unicode_symbol() for i in pieces])

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