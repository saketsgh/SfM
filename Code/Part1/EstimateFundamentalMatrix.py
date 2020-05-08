import numpy as np


def estimate_f_matrix(points8):

    A = []
    # points_a = points8[:, 0:2]
    # points_b = points8[:, 2:4]


    for points in points8:

        # get correspondences
        x1, y1, x2, y2 = points[0], points[1], points[2], points[3]


        # form the A matrix
        # A.append([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])
        A.append([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])

    A = np.array(A)
    # Least sqauares solution of AX = 0
    u, s, vt = np.linalg.svd(A)

    # last column of v is the solution
    v = vt.T
    x = v[:, -1]

    # enforce rank 2 constraint
    # F = np.reshape(x, (3, 3)).T
    F = np.reshape(x, (3, 3))
    U, S, VT = np.linalg.svd(F)
    S[2] = 0.0
    F = U.dot(np.diag(S)).dot(VT)


    return F
