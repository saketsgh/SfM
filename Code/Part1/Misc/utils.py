import os
import numpy as np
import math

def get_pts_from_txt(path, file_name):

    os.chdir(path)

    file = open(file_name, 'r')
    content = file.readlines()

    pts_from_txt = []

    for line in content:
        x1, y1, x2, y2, r, g, b = line.split()

        pts_from_txt.append([np.float32(x1), np.float32(y1), np.float32(x2), np.float32(y2), int(r), int(g), int(b)])

    os.chdir('../../Code/Part1')
    return pts_from_txt


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):

    Rt = np.transpose(R)

    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R):

    # check if rot mat is correct
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])
