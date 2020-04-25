import numpy as np


def estimate_e_matrix(F, K):

    E = np.dot(np.dot(K.T, F), K)

    # svd calc
    u, s, vt = np.linalg.svd(E)

    # singular value correction
    s = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])

    # recompute E
    E = np.dot(np.dot(u, s), vt)
    
    return E
