"""Unannotated, synthetic sign cards for checking the local recognition pipeline.

These illustrations are test inputs, not validation images from real roads.
"""

import cv2
import numpy as np


def speed_card(speed, unit="", size=320):
    card = np.full((size, size, 3), 230, dtype=np.uint8)
    centre = (size // 2, size // 2)
    radius = int(size * .42)
    cv2.circle(card, centre, radius, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(card, centre, radius, (0, 0, 220), int(size * .045), cv2.LINE_AA)
    text = str(speed)
    if unit:
        scale_num = size / 160
        (w_num, h_num), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale_num, 6)
        cv2.putText(card, text, ((size - w_num) // 2, int(size * 0.48)), cv2.FONT_HERSHEY_SIMPLEX,
                    scale_num, (10, 10, 10), 6, cv2.LINE_AA)
        scale_unit = size / 260
        (w_u, h_u), _ = cv2.getTextSize(unit, cv2.FONT_HERSHEY_SIMPLEX, scale_unit, 4)
        cv2.putText(card, unit, ((size - w_u) // 2, int(size * 0.72)), cv2.FONT_HERSHEY_SIMPLEX,
                    scale_unit, (10, 10, 10), 4, cv2.LINE_AA)
    else:
        scale = size / (120 if speed < 100 else 150)
        (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 7)
        cv2.putText(card, text, ((size - width) // 2, (size + height) // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (10, 10, 10), 7, cv2.LINE_AA)
    return card


def crossing_card(size=320):
    card = np.full((320, 320, 3), 230, dtype=np.uint8)
    cv2.rectangle(card, (25, 25), (295, 295), (175, 80, 0), -1)
    cv2.fillPoly(card, [np.array([[160, 45], [45, 270], [275, 270]])], (255, 255, 255))
    for x in range(80, 242, 35):
        cv2.fillPoly(card, [np.array([[x, 245], [x+14, 245], [x+4, 259], [x-10, 259]])], (15, 15, 15))
    black = (10, 10, 10)
    cv2.circle(card, (169, 124), 14, black, -1, cv2.LINE_AA)
    for a, b, width in [((164, 144), (148, 190), 18), ((158, 158), (129, 179), 10),
                         ((129, 179), (108, 179), 9), ((163, 150), (190, 178), 10),
                         ((190, 178), (207, 186), 9), ((149, 190), (181, 232), 13),
                         ((149, 190), (137, 212), 13), ((137, 212), (113, 235), 12)]:
        cv2.line(card, a, b, black, width, cv2.LINE_AA)
    return cv2.resize(card, (size, size))


def sample_board():
    board = np.full((430, 1720, 3), 230, dtype=np.uint8)
    cards = [(speed_card(40), "40 km/h"), (speed_card(50, "kmph"), "50 kmph"),
             (speed_card(60), "60 km/h"), (speed_card(70), "70 km/h"),
             (crossing_card(), "Pedestrian crossing")]
    for index, (card, caption) in enumerate(cards):
        x = 10 + index * 340
        board[55:375, x:x+320] = card
        cv2.putText(board, caption, (x+12, 408), cv2.FONT_HERSHEY_SIMPLEX, .7, (30, 30, 30), 2)
    return board
