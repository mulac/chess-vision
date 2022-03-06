import numpy as np
import cv2
import jenkspy

from .aruco import detect, DetectionError

SIZE = 800
MARGIN = 0


def label(pgn_game, corners, images, flipped=False, skip_moves=2, size=SIZE, margin=MARGIN):
    skip(images, skip_moves)
    for move, img in zip(pgn_game.mainline(), images):
        img = img["color"] if not flipped else np.flip(img["color"])
        yield label_move(move.board(), get_board(img, corners, size, margin), move.move.to_square, size, margin)


def label_occupied(pgn_game, corners, images, flipped=False, skip_moves=2, size=SIZE, margin=MARGIN):
    skip(images, skip_moves)
    for move, img in zip(pgn_game.mainline(), images):
        img = img["depth"] if not flipped else np.flip(img["depth"])
        yield label_move(move.board(), get_board(img, corners, size, margin), move.move.to_square, size, margin)


def skip(iterator, i):
    for _ in range(i):
        next(iterator)


def find_corners(images):
    for img in images:
        try:
            return get_corners(img["color"])
        except DetectionError:
            continue

    raise DetectionError("failed to find markers from any move")


def get_corners(image):
    return np.array([c[1][0][0] for c in detect(image)])


def label_move(pgn_board, board_img, square, size=SIZE, margin=MARGIN):
    piece_img = get_square(square, board_img, size, margin)
    label = pgn_board.piece_at(square)
    return piece_img, label


def get_board(img, corners, size=SIZE, margin=MARGIN):
    dest = np.float32([
        [margin, margin],
        [size+margin, margin],
        [margin, size+margin],
        [size+margin, size+margin]
    ])

    transform = cv2.getPerspectiveTransform(corners, dest)
    return cv2.warpPerspective(img, transform, (size+2*margin, size+2*margin))


def get_occupied_squares(depth_img, corners, size=SIZE, margin=MARGIN):
    depth_squares = get_squares(get_board(depth_img, corners), size=size, margin=margin)
    jnb = jenkspy.JenksNaturalBreaks(2)
    jnb.fit([np.sum(square) for square in depth_squares])
    # return [i for i, square in enumerate(depth_squares) if not jnb.predict(np.sum(square))]
    occupied_squares = np.where(np.logical_not(jnb.labels_))[0]
    return occupied_squares.tolist()


def get_square(square, img, size=SIZE, margin=MARGIN):
    squareSIZE = size // 8
    i = 7 - square % 8
    j = 7 - square // 8
    top_x = i * squareSIZE
    bot_x = (i+1)*squareSIZE + 2*margin
    top_y = j*squareSIZE
    bot_y = (j+1)*squareSIZE + 2*margin
    return img[top_y:bot_y, top_x:bot_x]


def get_squares(img, size=SIZE, margin=MARGIN):
    squareSIZE = size // 8
    squares = []
    for i in reversed(range(8)):
        top_y = i*squareSIZE
        bot_y = (i+1)*squareSIZE + 2*margin
        for j in reversed(range(8)):
            top_x = j*squareSIZE
            bot_x = (j+1)*squareSIZE + 2*margin
            squares.append(img[top_y:bot_y, top_x:bot_x])
    return squares
