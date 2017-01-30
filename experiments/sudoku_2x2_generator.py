import numpy as np
import csv
from random import randrange, sample

import datetime


def get_box(row, col):
    box_row = row // 2
    box_col = col // 2
    box = box_row * 2 + box_col
    return box

iteration = 0


def generate():
    board = np.zeros((4, 4))
    rows = [set() for i in range(4)]
    cols = [set() for i in range(4)]
    boxes = [set() for i in range(4)]
    try:
        for row in range(4):
            for col in range(4):
                box = get_box(row, col)
                options = {1, 2, 3, 4}
                options.difference_update(rows[row], cols[col], boxes[box])
                num = sample(options, 1).pop()
                board[row][col] = num
                rows[row].add(num)
                cols[col].add(num)
                boxes[box].add(num)
    except ValueError as ve:
        return generate()
    return board


def remove_nums(board, n=2):
    i = 0
    while i < n:
        x, y = randrange(4), randrange(4)
        if board[x][y] != 0:
            i += 1
            board[x][y] = 0


def convert_to_classes(board):
    class_board = np.zeros((4, 4, 4))
    for row in range(4):
        for col in range(4):
            num = int(board[row][col] - 1)
            if num != -1:
                class_board[row][col][num] = 1
    return class_board


def convert_to_normal(board):
    normal_board = np.zeros((4, 4))
    for row in range(4):
        for col in range(4):
            normal_board[row][col] = np.argmax(board[row][col]) + 1
    return normal_board


if __name__ == '__main__':
    n = 10000
    boards = np.zeros((n, 4**3))
    answer_boards = np.zeros((n, 4**3))
    t1 = datetime.datetime.now()
    for i in range(n):
        # each iteration takes about 2.82 for generation/removal, 3.57 with saving
        iteration += 1
        board = generate()
        answer_boards[i] = convert_to_classes(board).reshape((4 ** 3))
        remove_nums(board)
        class_board = convert_to_classes(board)
        boards[i] = class_board.reshape((4**3))
    t2 = datetime.datetime.now()
    print(t2 - t1)
