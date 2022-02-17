
import numpy as np
import pickle
import cv2


def main():
    game = f'games/{input("Enter game name: ")}.pkl'

    with open(game, "rb") as pkl_wb_obj:
        moves = pickle.load(pkl_wb_obj)

    for i, move in enumerate(moves):
        print(i)
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
    main()