import numpy as np


def anti_sym_mat(p):

    chi = np.array([[0, -p[2], p[1]],
                    [p[2], 0, -p[0]],
                    [-p[1], p[0], 0]])

    return chi



def linear_triagulation(C, R, K, inliers):

    # first camera pose
    M1 = np.identity(4)
    # make 3X4
    M1 = M1[0:3, :]
    # dot product with K to turn it into projection matrix of first camera
    M1 = np.dot(K, M1)

    # extract image points
    pts_1 = inliers[:, 0:2]
    pts_2 = inliers[:, 2:4]

    # make homog
    ones = np.ones((pts_1.shape[0], 1))
    pts_1 = np.hstack((pts_1, ones))
    pts_2 = np.hstack((pts_2, ones))


    M2 = np.hstack((R, -C))
    M2 = np.dot(K, M2)

    X_list = []

    for p1, p2 in zip(pts_1, pts_2):
        p1_chi = anti_sym_mat(p1)
        p2_chi = anti_sym_mat(p2)

        a1 = np.dot(p1_chi, M1)
        a2 = np.dot(p2_chi, M2)
        A = np.vstack((a1, a2))

        u, s, vt = np.linalg.svd(A)
        X = vt[:, -1]

        # non-homg
        X = X/X[3]
        X = X[:3]
        X = np.array(X)
        X = X.reshape((3, 1))

        X_list.append(X)

    return X_list
