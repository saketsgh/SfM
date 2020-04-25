import numpy as np
from LinearTriangulation import linear_triagulation


def cheirality_check(C, R, X_list):

    count = 0

    # third row of R
    r3 = R[2]

    for X in X_list:
        if(np.dot(r3, X-C) > 0):
            count += 1

    return count


def disambiguate_camera_pose(C_list, R_list, K, inliers):

    max_count = 0
    # for each Rot and Translation
    for R, C in zip(R_list, C_list):

        C = C.reshape((3, 1))

        # list of 3D points for the inliers
        X_list = linear_triagulation(C, R, K, inliers)

        # number of 3D points satisfying cheirality
        count = cheirality_check(C, R, X_list)
        print(count)
        print("\n")
        if(count > max_count):
            max_count = count
            R_best = R
            C_best = C

    return R_best, C_best
