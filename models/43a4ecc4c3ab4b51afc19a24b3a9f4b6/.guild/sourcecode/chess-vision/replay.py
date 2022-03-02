
import argparse
import pickle
import numpy as np
import cv2


def main(args):
    game = f'{args.dir}/{args.game_name}.pkl'

    with open(game, "rb") as pkl_wb_obj:
        while True:
            move = pickle.load(pkl_wb_obj)
            color_image, depth_image = move['color'], move['depth']
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape
            
            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)

            images = np.hstack((color_image, depth_colormap))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            cv2.waitKey(1)
            input()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Will replay a recorded chess game, displaying each move to the screen.')
    parser.add_argument('game_name', type=str,
                        help='the name of the game to replay')
    parser.add_argument('--dir', type=str, metavar='directory', default='games',
                        help='the directory the game file can be found in')

    main(parser.parse_args())