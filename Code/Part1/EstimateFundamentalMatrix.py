import numpy as np


def estimate_f_matrix(points8):

    A = []
    for points in points8:

        # get correspondences
        x1, y1, x2, y2 = points[0], points[1], points[2], points[3]


        # form the A matrix
        A.append([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])
        # A.append([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])


    A = np.array(A)

    # Least sqauares solution of AX = 0
    u, s, vt = np.linalg.svd(A)

    # last column of v is the solution
    v = vt.T
    x = v[:, -1]

    F = np.reshape(x, (3, 3))
    # u1, s1, vt1 = np.linalg.svd(F)

    # s2 = np.array([[s1[0], 0, 0], [0, s1[1], 0], [0, 0, 0]]) # Constraining Fundamental Matrix to Rank 2
    # F = np.dot(u1, np.dot(s2, vt1))

    # enforce rank 2 constraint
    F[2][2] = 0

    return F
