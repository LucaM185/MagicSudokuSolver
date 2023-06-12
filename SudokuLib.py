import cv2
import matplotlib.pyplot as plt
import numpy as np
from numba import njit

def blurFast(img, d=5):
    return cv2.filter2D(img, -1, np.zeros((d, d)) + 1 / d ** 2)

@njit
def convolve2d(image, kernel):
    # Get the dimensions of the image and kernel
    image_rows, image_cols = image.shape
    kernel_rows, kernel_cols = kernel.shape

    # Determine the size of the output image
    output_rows = image_rows - kernel_rows + 1
    output_cols = image_cols - kernel_cols + 1

    # Initialize the output image with zeros
    output = np.zeros((output_rows, output_cols))

    # Slide the kernel over the image
    for i in range(output_rows):
        for j in range(output_cols):
            # Multiply the overlapping pixels with the kernel
            output[i][j] = np.sum(image[i:i + kernel_rows, j:j + kernel_cols] * kernel)

    return normalize(np.abs(output))


@njit
def normalize(arr):
    # Compute the minimum and maximum values of the array
    min_val = np.min(arr)
    max_val = np.max(arr)

    # Normalize the array by subtracting the minimum value and dividing by the range
    return (arr - min_val) / (max_val - min_val)


from numba import njit


@njit()
def findLines(img):
    w, h = img.shape[:2]
    out = np.zeros((w, h))
    lastCol = 0
    for y in range(h):
        c = 0
        startingX = 0
        for x in range(w):
            if img[x, y] == lastCol and img[x, y] > 0.5:
                c += 1
            else:
                out[startingX:x, y] += c  # find long lines
                startingX = x
                c = 0
            lastCol = img[x, y]

    # Repeat flipped
    lastCol = 0
    for x in range(w):
        c = 0
        startingY = 0
        for y in range(h):
            if img[x, y] == lastCol and img[x, y] > 0.5:
                c += 1
            else:
                # TODO CHECK IF INTERSECTING WITH AT LEAST 4 DISTINCT COLUMNS
                out[x, startingY:y] *= c  # find the intersections with the other lines
                startingY = y
                c = 0
            lastCol = img[x, y]
    return out


@njit
def blur(img, d=5):
    k = np.zeros((d, d)) + 1 / d ** 2
    return convolve2d(img, k)


def ffv(l):  # find first value
    m, M = None, None
    for n, v in enumerate(l):
        if v > 0:
            if m == None:
                m = n
        if v == 0:
            if m != None and M == None:
                M = n
                return (m + M) // 2


def boundries(img):
    if len(img.shape) == 3: img = img[:, :, 1]
    img = np.float32(img)
    h, w = img.shape

    img = np.abs(img[1:] - img[:-1])[:, :-1] + np.abs(img[:, 1:] - img[:, :-1])[
                                               :-1]  # detect difference in pixel brightness
    img = blur(img, 7)
    img[img > 0.2] = 1
    lines = findLines(img)
    lines = blur(lines, 15)

    rows = np.sum(lines, axis=0)
    rows[rows < np.max(rows) / 2] = 0
    cols = np.sum(lines, axis=1)
    cols[cols < np.max(cols) / 2] = 0

    coords = [ffv(rows), w - ffv(reversed(rows)), ffv(cols), h - ffv(reversed(cols))]
    return coords


def checkDoubles(s):  # TRUE if FOUND
    list = []
    for element in s:
        if element in list and element != 0:
            return True
        else:
            list.append(element)


def checkBoxes(s):
    s = np.array(s)
    for i in range(3):
        for j in range(3):
            if checkDoubles(s[i * 3:i * 3 + 3, j * 3:j * 3 + 3].flatten().tolist()):
                return False
    return True


def checkColumns(s):
    for i in range(9):
        if checkDoubles(s[i]):
            return False
    return True


def checkRows(s):
    for i in range(9):
        if checkDoubles(np.array(s)[:, i].tolist()):
            return False
    return True


def check(s):
    try:
        s = s.tolist()
    except:
        pass

    if checkBoxes(s) and checkColumns(s) and checkRows(s):
        return True  # OK
    else:
        return False


def possibilities(m):
    m = np.array(m)
    poss = []
    test = m.copy()
    for i in range(9):
        poss.append([])
        test = m.copy()
        for j in range(9):
            poss[-1].append([])
            test = m.copy()
            if test[i][j] == 0:
                for test[i][j] in range(1, 10):
                    if check(test):
                        poss[-1][-1].append(test[i][j])
    return poss


def solve(m, poss):
    p = np.argmin(np.array(m))
    pos = [p // 9, p % 9]
    board = m.copy()
    for i in poss[pos[0], pos[1]]:
        if i.size == 0: break
        board[pos[0], pos[1]] = i
        if check(board):
            if np.min(board) == 1:  # To make it work make 80 the last zero
                return board
            solution = solve(board.copy(), poss)
            if np.min(solution) == 1: return solution
