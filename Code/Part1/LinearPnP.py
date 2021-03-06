import numpy as np

def linear_pnp(new_img_2d_3d, K):

    # A = []
    A = np.empty((0, 12), np.float32)
    for p in new_img_2d_3d:

        # image coordinates
        x, y = p[0], p[1]

        # normalse the image points
        normalised_pts = np.dot(np.linalg.inv(K), np.array([[x], [y], [1]]))
        normalised_pts = normalised_pts/normalised_pts[2]

        # corresp 3d coordinates
        X = p[2:]
        X = X.reshape((3, 1))

        # convert to homog
        X = np.append(X, 1)

        zeros = np.zeros((4,))
        #
        A_1 = np.hstack((zeros, -X.T, normalised_pts[1]*(X.T)))
        A_2 = np.hstack((X.T, zeros, -normalised_pts[0]*(X.T)))
        A_3 = np.hstack((-normalised_pts[1]*(X.T), normalised_pts[0]*X.T, zeros))

        # A_1 = [X[0], X[1], X[2], 1, 0, 0, 0, 0, -normalised_pts[0]*X[0], -normalised_pts[0]*X[1], -normalised_pts[0]*X[2], -normalised_pts[0]]
        # A_2 = [0, 0, 0, 0, X[0], X[1], X[2], 1, -normalised_pts[1]*X[0], -normalised_pts[1]*X[1], -normalised_pts[1]*X[2], -normalised_pts[1]]

        for a in [A_1, A_2, A_3]:
			A = np.append(A, [a], axis=0)
        # for a in [A_1, A_2]:
		# 	A = np.append(A, [a], axis=0)

    # A = A.reshape((A.shape[0], -1))
    A = np.float32(A)
    U, S, VT = np.linalg.svd(A)

    V = VT.T

    # last column of v is the solution i.e the required pose
    pose = V[:, -1]
    pose = pose.reshape((3, 4))

    # extract rot and trans
    R_new = pose[:, 0:3]
    T_new = pose[:, 3]
    R_new = R_new.reshape((3, 3))
    T_new = T_new.reshape((3, 1))

    # impose orthogonality constraint
    U, S, VT = np.linalg.svd(R_new)
    R_new = np.dot(U, VT)

    # check det sign
    if np.linalg.det(R_new) < 0:
        R_new = -R_new
        T_new = -T_new
    # print(R_new)
    C_new = -np.dot(R_new.T, T_new)

    return R_new, C_new
