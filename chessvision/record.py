""" Provides a realsense camera abstraction for recording games """

import argparse
import pickle
import chess
import cv2

from .camera import RealsenseCamera


class Recorder:
    def __init__(self, config, camera):
        self.config, self.camera = config, camera
        
    def game(self):
        picklefile = f'{self.config.dir}/{self.config.file_name}.pkl'
        self.move = 0
        
        def record(color, depth):
            cv2.imwrite("data/current.jpg", color.copy())
            pickle.dump({"color": color.copy(), "depth": depth.copy()}, pkl_file)
            input(f"\n{self.move}: waiting...")
            self.move += 1
            
        try:
            pkl_file = open(picklefile, "wb")
            input("Press [ENTER] to begin:")
            self.camera.loop(record)
        except KeyboardInterrupt:
            print("\nsaving pickle file...")
        finally:
            self.camera.close()
            pkl_file.close()


    def video(self):
        videofile = f'{self.config.dir}/{self.config.file_name}.mp4'
        writer = cv2.VideoWriter(videofile, cv2.VideoWriter_fourcc(*'DIVX'), 10, self.camera.resolution)

        def record(colour, _):
            writer.write(colour)
            cv2.imshow('recorder', colour)
            cv2.waitKey(1)

        try:
            self.camera.loop(record)
        except KeyboardInterrupt:
            print(f"saving video file to {videofile}")
        finally:
            self.camera.close()
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
        Recorder(args, RealsenseCamera()).video()
    else:
        Recorder(args, RealsenseCamera()).game()