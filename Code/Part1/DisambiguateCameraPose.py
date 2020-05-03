import numpy as np
from LinearTriangulation import linear_triagulation
import matplotlib.pyplot as plt


def cheirality_check(C, R, X_list):

    count = 0

    # third row of R
    r3 = R[2]

    for X in X_list:
        if(np.dot(r3, X-C) > 0):
            count += 1

    return count


def plot_points(X_list, i):

    # plot the points
    X_list = np.array(X_list)

    # reshape
    X_list = X_list.reshape((X_list.shape[0], -1))

    # extract x and z
    x = X_list[:, 0]
    z = X_list[:, 2]

    colormap = np.array(['y', 'b', 'k', 'r'])
    # markers = np.array(['*', 'x', 'D', 'o'])

    ax = plt.gca()
    # ax.scatter(x, z, c=colormap[i], marker=markers[i], alpha=0.5)
    ax.scatter(x, z, s=3, color = colormap[i])

def disambiguate_camera_pose(C_list, R_list, K, inliers):

    # to keep track of the maximum X points
    max_count = 0

    # for plotting
    i = 0
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # for each Rot and Translation
    for R, C in zip(R_list, C_list):

        C = C.reshape((3, 1))

        # list of 3D points for the inliers
        X_list = linear_triagulation(C, R, K, inliers)

        # number of 3D points satisfying cheirality
        count = cheirality_check(C, R, X_list)

        plot_points(X_list, i)

        if(count > max_count):
            max_count = count
            R_best = R
            C_best = C
            X_list_best = X_list

        i+=1
    plt.xlim(-10, 10)
    plt.ylim(-15, 15)
    plt.show()
    return R_best, C_best, X_list_best
