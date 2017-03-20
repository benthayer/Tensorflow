import numpy as np
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


def is_valid(board):
    rows = [set() for i in range(4)]
    cols = [set() for i in range(4)]
    boxes = [set() for i in range(4)]
    board = board.copy()
    board.shape = 4, 4
    for row in range(4):
        for col in range(4):
            box = get_box(row, col)
            options = {1, 2, 3, 4}
            options.difference_update(rows[row], cols[col], boxes[box])
            num = board[row][col]
            if num == 0:
                continue
            if num not in options:
                return False
            rows[row].add(num)
            cols[col].add(num)
            boxes[box].add(num)
    return True


def gen_all():
    # This function generates all possible boards and returns them in a big list
    # The code is pretty inefficient (2.26ms / board), but there's only 288 2x2 boards
    boards = np.zeros((0, 16))
    board = np.zeros(16)
    i = 0
    while True:
        if i == 16:
            boards = np.append(boards, [board], axis=0)
            i -= 1
        if board[i] == 4:
            if i == 0:
                break
            board[i] = 0
            i -= 1
            continue
        board[i] += 1
        if is_valid(board):
            i += 1
    return boards


def remove_nums(board, n=7):
    new_board = board.copy()
    new_board.shape = 4, 4
    shape = board.shape
    i = 0
    while i < n:
        x, y = randrange(4), randrange(4)
        if new_board[x][y] != 0:
            i += 1
            new_board[x][y] = 0
    new_board.shape = shape
    return new_board


def convert_to_classes(board):
    board = board.copy()
    board.shape = 4, 4
    class_board = np.zeros((4, 4, 4))
    for row in range(4):
        for col in range(4):
            num = int(board[row][col] - 1)
            if num != -1:
                class_board[row][col][num] = 1
    class_board.shape = 4**3
    return class_board


def convert_to_normal(board, ones=False):
    normal_board = np.zeros((4, 4))
    for row in range(4):
        for col in range(4):
            if ones is False:
                normal_board[row][col] = np.argmax(board[row][col]) + 1
            else:
                num = np.where(board[row][col] == 1)[0]
                if len(num) == 0:
                    normal_board[row][col] = 0
                else:
                    normal_board[row][col] = num + 1
    return normal_board


def get_training_and_test_sets(num_to_remove=7, pct_training=0.7):
    unformatted_solutions = gen_all()
    num_puzzles = len(unformatted_solutions)
    np.random.shuffle(unformatted_solutions)

    puzzles = np.zeros((len(unformatted_solutions), 4**3))
    solutions = np.zeros(puzzles.shape)

    for i, solution in enumerate(unformatted_solutions):
        puzzles[i] = convert_to_classes(remove_nums(solution, num_to_remove))
        solutions[i] = convert_to_classes(solution)

    split = int(pct_training * num_puzzles)

    training_puzzles = puzzles[:split]
    training_solutions = solutions[:split]
    test_puzzles = puzzles[split:]
    test_solutions = solutions[split:]

    return training_puzzles, training_solutions, test_puzzles, test_solutions


if __name__ == '__main__':
    t1 = datetime.datetime.now()
    gen_all()
    print((datetime.datetime.now() - t1)/288)