
import argparse
import pickle
import numpy as np
import cv2


def main(args):
    game = f'{args.dir}/{args.game_name}.pkl'

    with open(game, "rb") as pkl_wb_obj:
        i = 0
        while True:
            try:
                move = pickle.load(pkl_wb_obj)
            except StopIteration:
                break

            color_image, depth_image = move['color'], move['depth']
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape
            print(i, color_colormap_dim, depth_colormap_dim)
            
            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                # color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                color_image = translate_img(color_image, depth_colormap)

            images = np.hstack((color_image, depth_colormap))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', cv2.resize(images, (1100, 350)))
            cv2.waitKey(1)
            input()
            i += 1


def translate_img(src_img, dest_img):
    x, y = src_img.shape[1], src_img.shape[0]
    xd, yd = dest_img.shape[1], dest_img.shape[0]

    src = np.float32([
        [0, 0],
        [x, 0],
        [0, y],
        [x, y]
    ])

    dest = np.float32([
        [0, 0],
        [xd, 0],
        [0, yd],
        [xd, yd]
    ])


    transform = cv2.getPerspectiveTransform(src, dest)
    return cv2.warpPerspective(src_img, transform, (xd, yd))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Will replay a recorded chess game, displaying each move to the screen.')
    parser.add_argument('game_name', type=str,
                        help='the name of the game to replay')
    parser.add_argument('--dir', type=str, metavar='directory', default='games',
                        help='the directory the game file can be found in')

    main(parser.parse_args())