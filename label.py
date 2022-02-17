import argparse
import pickle
import chess.pgn
import numpy as np
import cv2
import aruco


def get_corners(moves):
    for move in moves:
        try:
            return [c[1] for c in aruco.detect(move["color"])]
        except aruco.DetectionError:
            continue

    raise aruco.DetectionError("failed to find markers from any move")


def get_squares(img, size, margin):
    square_size = size // 8
    squares = []
    for i in range(8):
        top_x = i*square_size
        bot_x = (i+1)*square_size + 2*margin
        for j in range(8):
            top_y = j*square_size
            bot_y = (j+1)*square_size + 2*margin
            squares.append(img[top_y:bot_y, top_x:bot_x])
    return squares


def get_square(square, img, size, margin):
    square_size = size // 8
    i = square % 8
    j = 7 - square // 8
    top_x = i*square_size
    bot_x = (i+1)*square_size + 2*margin
    top_y = j*square_size
    bot_y = (j+1)*square_size + 2*margin
    return img[top_y:bot_y, top_x:bot_x]


def get_board(img, corners, size, margin):
    dest = np.float32([
        [margin,size+margin], 
        [size+margin,size+margin], 
        [margin,margin], 
        [size+margin, margin]
    ])

    transform = cv2.getPerspectiveTransform(corners, dest)
    return cv2.warpPerspective(img, transform, (size+2*margin, size+2*margin))


def label_move(move, img, corners, size=800, margin=0):
    board_img = get_board(img, corners, size, margin)
    piece_img = get_square(move.move.to_square, board_img, size, margin)
    label = move.board().piece_at(move.move.to_square)
    return piece_img, label


def label(path, game_num):
    with open(f"{path}.pgn") as pgn:
        for i in range(game_num):
            chess.pgn.skip_game(pgn)
        game = chess.pgn.read_game(pgn)

    with open(f"{path}_{game_num}.pkl", "rb") as pkl_wb_obj:
        recorded_moves = pickle.load(pkl_wb_obj)

    board = np.array([corner[0][0] for corner in get_corners(recorded_moves)])

    images = []
    labels = []
    for move, img in zip(game.mainline(), recorded_moves):
        img, lbl = label_move(move, img["color"], board, size=800, margin=10)
        images.append(img)
        labels.append(lbl)

    return images, labels
