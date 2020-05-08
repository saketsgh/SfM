import numpy as np

def linear_pnp(new_img_2d_3d, K):

    A = []
    for p in new_img_2d_3d:

        # image coordinates
        x, y = p[0], p[1]

        # normalse the image points
        normalised_pts = np.dot(np.linalg.inv(K), np.array([[x], [y], [1]]))

        # corresp 3d coordinates
        X = p[2:]
        X = X.reshape((3, 1))

        # convert to homog
        X = np.append(X, 1)

        zeros = np.zeros((1, 4))
        A.append([zeros.flatten(), -X.T.flatten(), normalised_pts[1]*(X.T).flatten()])
        A.append([X.T.flatten(), zeros.flatten(), -normalised_pts[0]*(X.T).flatten()])
        A.append([-normalised_pts[1]*(X.T).flatten(), normalised_pts[0]*X.T.flatten(), zeros.flatten()])

    A = np.array(A)
    A = A.reshape((A.shape[0], -1))
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

    C_new = -np.dot(R_new.T, T_new)

    return R_new, C_new
