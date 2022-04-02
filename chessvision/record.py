""" Provides a realsense camera abstraction for recording games """

import argparse
import pickle
import cv2

from .camera import RealsenseCamera


def game(args):
    picklefile = f'{args.dir}/{args.file_name}.pkl'
    camera=RealsenseCamera()
    def record(color, depth):
        cv2.imwrite("data/current.jpg", color.copy())
        pickle.dump({"color": color.copy(), "depth": depth.copy()}, pkl_file)
        input("\nwaiting...")
        
    try:
        pkl_file = open(picklefile, "wb")
        input("Press [ENTER] to begin:")
        camera.loop(record)
    except KeyboardInterrupt:
        print("\nsaving pickle file...")
    finally:
        camera.close()
        pkl_file.close()


def video(args, camera=RealsenseCamera()):
    videofile = f'{args.dir}/{args.file_name}.mp4'
    writer = cv2.VideoWriter(videofile, cv2.VideoWriter_fourcc(*'DIVX'), 10, camera.resolution)

    def record(colour, _):
        writer.write(colour)
        cv2.imshow('recorder', colour)
        cv2.waitKey(1)

    try:
        camera.loop(record)
    except KeyboardInterrupt:
        print(f"saving video file to {videofile}")
    finally:
        camera.close()
        writer.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Will record a recorded chess game, saving it to disk.'
                    '[ENTER] will take a screenshot.')
    parser.add_argument('file_name', type=str,
        help='the file_name to save the game as')
    parser.add_argument('--video', action='store_true', default=False,
        help='just record a .mp4 and not individual frames per move')
    parser.add_argument('--dir', type=str, metavar='directory', default='games',
        help='the directory the game file can be found in')

    if (args := parser.parse_args()).video:  
        video(args)
    else:
        game(args)