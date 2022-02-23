import os
import enum
import pickle
import itertools
import argparse
import numpy as np
import cv2


markers = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

class Marker(enum.IntEnum):
    WHITE_QUEEN = 0
    WHITE_KING = 1
    BLACK_QUEEN = 2
    BLACK_KING = 3

    @classmethod
    def ids(cls):
        return (member.value for member in cls)


class DetectionError(Exception):
    pass


_detection_params = cv2.aruco.DetectorParameters_create()
_detection_params.polygonalApproxAccuracyRate = 0.02
_detection_params.minMarkerPerimeterRate = 0.05
_detection_params.maxMarkerPerimeterRate = 0.25
_detection_params.adaptiveThreshConstant = 7
_detection_params.adaptiveThreshWinSizeMin = 7
_detection_params.adaptiveThreshWinSizeMax = 99
_detection_params.adaptiveThreshWinSizeStep = 2
_detection_params.minDistanceToBorder = 1
# _detection_params.minOtsuStdDev = 7
# _detection_params.perspectiveRemovePixelPerCell = 10
# _detection_params.perspectiveRemoveIgnoredMarginPerCell = 0.05
# _detection_params.maxErroneousBitsInBorderRate = 0.55
_detection_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX


def generate(dir="images/markers", size=128):
    if not os.path.isdir(dir):
        raise FileNotFoundError(f'directory "{dir}" not found')
    for marker in Marker:
        img = cv2.aruco.drawMarker(markers, marker.value, size)
        cv2.imwrite(f'{dir}/{marker.name}.png', img)


def detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    corners, ids, rejectedCorners = cv2.aruco.detectMarkers(blurred, markers, parameters=_detection_params)
    if ids is None:
        raise DetectionError(f"failed to detect any markers")
    if len(ids) != len(Marker) or not all(m in itertools.chain.from_iterable(ids) for m in Marker.ids()):
        raise DetectionError(f"detected {ids}, needed {Marker.ids()}")
    return sorted((Marker(ids[i]), corners[i]) for i in range(0, 4))


def _detect(path="images/markers/test.jpg", out="images/markers/test_out.jpg"):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255,
	cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 99, _detection_params.adaptiveThreshConstant)
    cv2.imwrite("test_out_thresh.jpg", thresh)

    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(blurred, markers, parameters=_detection_params)
    print(f'found {ids} and got {len(rejectedImgPoints)} rejected points')
    cv2.imwrite(out, cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids))
    cv2.imwrite("test_out_fail.jpg", cv2.aruco.drawDetectedMarkers(img.copy(), rejectedImgPoints))


def detect_pkl(args):
    game = f'{args.dir}/{args.game_name}.pkl'

    with open(game, "rb") as pkl_wb_obj:
        moves = pickle.load(pkl_wb_obj)

    img = moves[args.move]["color"]

    cv2.imwrite("test.jpg", img)
    _detect("test.jpg", "test_out.jpg")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Will attempt to detect aruco markers from a recorded chess game.')
    parser.add_argument('game_name', type=str,
                        help='the name of the game to get the image from')
    parser.add_argument('--move', type=int, default=0,
                        help='the move number of that game')
    parser.add_argument('--dir', type=str, metavar='directory', default='games',
                        help='the directory the game file can be found in')

    detect_pkl(parser.parse_args())