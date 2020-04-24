import numpy as np


def estimate_f_matrix(points8):

    A = []
    for points in points8:

        # get correspondences
        x1, y1, x2, y2 = points[0], points[1], points[2], points[3]


        # form the A matrix
        A.append([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])

    A = np.array(A)

    # Least sqauares solution of AX = 0
    u, s, vt = np.linalg.svd(A)

    x = vt[-1]

    F = np.reshape(x, (3, 3))

    # enforce rank 2 constraint
    F[2][2] = 0

    return F
