import os
import enum
import cv2


dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

class Marker(enum.Enum):
    WHITE_QUEEN = 0
    WHITE_KING = 1
    BLACK_QUEEN = 2
    BLACK_KING = 3

def generate(dir="images/markers", size=128):
    if not os.path.isdir(dir):
        raise FileNotFoundError(f'directory "{dir}" not found')
    for marker in Marker:
        img = cv2.aruco.drawMarker(dictionary, marker.value, size)
        cv2.imwrite(f'{dir}/{marker.name}.png', img)