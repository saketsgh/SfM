import numpy as np
from LinearTriangulation import linear_triagulation
import matplotlib as mpl
import matplotlib.pyplot as plt
import math


def cheirality_check(C, R, X_list):

    count = 0

    # third row of R
    r3 = R[2]

    for X in X_list:
        if(np.dot(r3, X-C) > 0):
            count += 1

    return count


def plot_points(X_list, i, C, R):

    # plot the points
    X_list = np.array(X_list)

    # reshape
    X_list = X_list.reshape((X_list.shape[0], -1))

    # extract x and z
    x = X_list[:, 0]
    z = X_list[:, 2]

    colormap = np.array(['y', 'b', 'c', 'r'])

    ax = plt.gca()

    # extract euler angles(w.r.t y since plot is x vs z) to plot markers
    ax.plot(0, 0, marker=mpl.markers.CARETDOWN, markersize=15, color = 'k')

    euler_angles = rotationMatrixToEulerAngles(R)
    angles_camera = np.rad2deg(euler_angles)

    # plot the cameras
    t = mpl.markers.MarkerStyle(marker=mpl.markers.CARETDOWN)
    t._transform = t.get_transform().rotate_deg(int(angles_camera[1]))

    # ax.plot(-C[0], -C[2], marker=(3, 0, int(angles_camera[1])), markersize=15, color=colormap[i])
    ax.scatter((-C[0]), (-C[2]), marker=t, s=250, color=colormap[i])
    ax.scatter(x, z, s=4, color=colormap[i])
    # ax.scatter(-x, -z, s=4, color=colormap[i])


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):

    Rt = np.transpose(R)

    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R):

    # check if rot mat is correct
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


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

        plot_points(X_list, i, C, R)

        if(count > max_count):
            max_count = count
            R_best = R
            C_best = C
            X_list_best = X_list
            index = i

        i+=1
    plt.xlim(-15, 15)
    plt.ylim(-20, 20)
    plt.show()
    return R_best, C_best, X_list_best, index
