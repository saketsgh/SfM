import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from LinearTriangulation import linear_triagulation
from Misc.utils import PlotFuncs


def cheirality_check(C, R, X_list):

    count = 0

    # third row of R
    r3 = R[2]

    for X in X_list:
        if(np.dot(r3, X-C) > 0):
            count += 1

    return count


def disambiguate_camera_pose(M1, C2_list, R2_list, K, inliers):

    # to keep track of the maximum X points
    max_count = 0

    # for plotting
    i = 0
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plot_funcs = PlotFuncs()

    print("plotting all the 4 obtained camera poses and their respective world points\n")
    print("cheirality counts -->")
    # for each Rot and Translation
    for R, C in zip(R2_list, C2_list):

        C = C.reshape((3, 1))

        # list of 3D points for the inliers
        X_list = linear_triagulation(M1, C, R, K, inliers)

        # number of 3D points satisfying cheirality
        count = cheirality_check(C, R, X_list)

        plot_funcs.plot_triangulated_points(X_list, i, C, R)
        print(count)
        if(count > max_count):
            max_count = count
            R_best = R
            C_best = C
            X_list_best = X_list
            index = i

        i+=1
    plt.xlim(-15, 20)
    plt.ylim(-30, 40)
    plt.show()
    print("\n")
    return R_best, C_best, X_list_best, index
