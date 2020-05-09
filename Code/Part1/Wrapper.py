import random
import os
import cv2
import numpy as np
from GetInliersRANSAC import get_inliers_ransac
from GetInliersRANSAC import get_pts_from_txt
from EssentialMatrixFromFundamentalMatrix import estimate_e_matrix
from ExtractCameraPose import extract_camera_pose
from DisambiguateCameraPose import disambiguate_camera_pose
from NonlinearTriangulation import nonlinear_triang
from LinearPnP import linear_pnp
import matplotlib as mpl
import matplotlib.pyplot as plt

from Misc.utils import*


def linear_vs_non_linear(X_linear, X_non_linear, index):

    # extract the x and the z components

    X_linear = np.array(X_linear)
    X_linear = X_linear.reshape((X_linear.shape[0], -1))

    x_l = X_linear[:, 0]
    z_l = X_linear[:, 2]

    x_indices = range(0, np.shape(X_non_linear)[0], 3)
    z_indices = range(2, np.shape(X_non_linear)[0], 3)
    x_nl = X_non_linear[x_indices]
    z_nl = X_non_linear[z_indices]

    # plot linear and non linear points and compare
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # define color scheme using index to identify which pose from the previous plot was correct
    colormap = np.array(['y', 'b', 'c', 'r'])

    ax.scatter(x_l, z_l, s=40, marker='+', color = colormap[index])
    ax.scatter(x_nl, z_nl, s=7, color = 'k')
    plt.xlim(-15, 20)
    plt.ylim(-30, 40)
    plt.show()


def plot_correspondences(imgA, imgB, inlier_locs, outlier_locs, matchesImg="matches12", save=False):

    # club image
    height = max(imgA.shape[0], imgB.shape[0])
    width = imgA.shape[1] + imgB.shape[1]
    clubimage = np.zeros((height, width, 3), type(imgA.flat[0]))
    clubimage[:imgA.shape[0], :imgA.shape[1], :] = imgA
    clubimage[:imgB.shape[0], imgA.shape[1]:, :] = imgB
    shiftX = imgA.shape[1]
    shiftX = np.float32(shiftX)

    # printing inliers
    for i, p in enumerate(inlier_locs):
        x1, y1 = p[0], p[1]
        x2, y2 = p[2], p[3]
        # print("\n")
        # print(p)
        cv2.circle(clubimage, (x1, y1), 3, (255, 0, 0), 1)
        cv2.circle(clubimage, (x2+shiftX, y2), 3, (255, 0, 0), 1)
        cv2.line(clubimage, (x1, y1), (x2+shiftX, y2), (0, 255, 0), 1)

    # # printing outliers
    for _, p in enumerate(outlier_locs):

        x1, y1 = p[0], p[1]
        x2, y2 = p[2], p[3]

        cv2.circle(clubimage, (x1, y1), 3, (0, 0, 0), 1)
        cv2.circle(clubimage, (x2+shiftX, y2), 3, (0, 0, 0), 1)
        cv2.line(clubimage, (x1, y1), (x2+shiftX, y2), (0, 0, 255), 1)

    cv2.namedWindow(matchesImg, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(matchesImg, (clubimage.shape[0]/2, clubimage.shape[0]/2))
    cv2.imshow(matchesImg, clubimage)
    cv2.waitKey(0)
    if(save):
        cv2.imwrite(matchesImg+".jpg", clubimage)
    cv2.destroyAllWindows()


def load_images(path):

    images = []

    for i in range(1, 7):
        img = cv2.imread(path+str(i)+".jpg")
        images.append(img)

    return images



def get_2d_corresp(ref_img_2d_3d, matches):

    new_img_2d_3d = []
    for pts in matches:
        matches_ref_img = pts[0:2]
        matches_new_img = pts[2:4]

        # compare with all other points in ref image to get
        ref_2d = ref_img_2d_3d[:, 0:2]
        ssd = ref_2d - matches_ref_img
        ssd = ssd**2
        ssd = np.sum(ssd, axis=1)
        locs = np.where(ssd == 0)[0]
        if(np.shape(locs)[0]):
            # we got the same point from inliers calculated in previous steps and from the matches
            new_img_2d_3d.append([matches_new_img[0], matches_new_img[1],
            ref_img_2d_3d[locs][0][2], ref_img_2d_3d[locs][0][3], ref_img_2d_3d[locs][0][4]])

    return np.array(new_img_2d_3d)



def plot_camera_poses(poses, ref_imgs):

    colormap = ['y', 'b', 'c', 'm', 'r', 'k']

    fig = plt.figure()
    ax = fig.add_subplot(111)


    for c, p in zip(colormap, poses):

        # extract the camera pose
        pose = poses[p]
        R = pose[:, 0:3]
        C = pose[:, 3:]

        # extract the 3d correspondences
        corresp_2d_3d = ref_imgs[p]
        X = corresp_2d_3d[:, 2:]

        # plot the cameras

        euler_angles = rotationMatrixToEulerAngles(R)
        angles_camera = np.rad2deg(euler_angles)

        t = mpl.markers.MarkerStyle(marker=mpl.markers.CARETDOWN)
        t._transform = t.get_transform().rotate_deg(int(angles_camera[1]))

        # ax.plot(-C[0], -C[2], marker=(3, 0, int(angles_camera[1])), markersize=15, color=colormap[i])
        ax.scatter((-C[0]), (-C[2]), marker=t, s=250, color=c)
        ax.scatter(X[:, 0], X[:, 2], s=4, color=c)
    plt.xlim(-15, 20)
    plt.ylim(-30, 40)
    plt.show()


def main():

    # path to load the data from
    path='../../Data/Data/'

    # loading images
    images = load_images(path)

    # defining image pairs
    file_names = ["matches12", "matches13",
                "matches14", "matches23",
                "matches24", "matches34",
                "matches35", "matches36",
                "matches45", "matches46", "matches56"]


    # given camera calibration matrix
    K = np.array([[568.996140852, 0, 643.21055941], [0, 568.988362396, 477.982801038], [0, 0, 1]])

    # define camera 1 as the world pose
    M1 = np.identity(4)
    # # make 3X4
    M1 = M1[0:3, :]
    # # dot product with K to turn it into projection matrix of first camera
    M1 = np.dot(K, M1)
    # M1 = M1.astype(float32)

    # for each image pairs compute F, E

    '''.............................get inliers and fundamental matrix using RANSAC...........................'''
    file_name=file_names[0]
    max_inliers_locs, min_outliers_locs, F_max_inliers, pts_left, pts_right = get_inliers_ransac(path, file_name)

    # plotting correspondences for inliers
    plot_correspondences(images[0], images[1], max_inliers_locs, min_outliers_locs, file_name)

    '''.............................essential matrix...........................'''
    E = estimate_e_matrix(F_max_inliers, K)
    C2_list, R2_list = extract_camera_pose(E)


    '''.............................disambiguate pose of cam 2...........................'''
    R2, C2, X_list, index = disambiguate_camera_pose(M1, C2_list, R2_list, K, max_inliers_locs)

    # construct pose of camera 2
    I = np.identity(3)
    M2 = np.hstack((I, -C2))
    M2 = np.dot(K, np.dot(R2, M2))

    '''.............................non linear triangulation...........................'''
    X_list_refined =  nonlinear_triang(M1, M2, X_list, max_inliers_locs, K)

    # compare non-linear triangulation with linear by plot
    linear_vs_non_linear(X_list, X_list_refined, index)

    # create a dict to store all the poses
    poses = {}

    # store the pose of 1st and 2nd cam
    poses[1] = np.identity(4)[0:3, :]
    poses[2] = np.dot(R2, np.hstack((np.identity(3), -C2)))

    '''.............................PnP to estimate remaining poses...........................'''
    # using correspondences between the following image pairs for PnP
    image_nums = [[2, 3], [3, 4], [4, 5], [5, 6]]

    # create a dict consisting of ref image 2d-3d correspondences
    ref_imgs = {}

    # first we need to get inliers of image i(3-6) wrt previously estimated camera pose so that we
    # match the 2D image point with the already calculated 3D point
    pts_ref_img = max_inliers_locs[:, 0:2]
    X_list_refined = np.reshape(X_list_refined, (pts_ref_img.shape[0], 3))
    ref_img_2d_3d = np.hstack((pts_ref_img, X_list_refined))
    ref_imgs[1] = ref_img_2d_3d

    # same thing for image 2
    pts_ref_img = max_inliers_locs[:, 2:4]
    ref_img_2d_3d = np.hstack((pts_ref_img, X_list_refined))
    ref_imgs[2] = ref_img_2d_3d

    # estimate pose for the remaining cams
    for _, nums in enumerate(image_nums):

        ref_img_num = nums[0]
        new_img_num = nums[1]

        file_name = "matches"+str(ref_img_num)+str(new_img_num)+".txt"
        print(file_name)

        # get the 2d-3d correspondences for the ref image
        ref_img_2d_3d = ref_imgs[ref_img_num]

        # next we must compare it with the points found using given matches
        matches = get_pts_from_txt(path, file_name)
        matches = np.array(matches, np.float32)
        # print(np.shape(matches))

        new_img_2d_3d = get_2d_corresp(ref_img_2d_3d, matches)
        print(np.shape(new_img_2d_3d))

        # add it to ref_imgs for future use
        ref_imgs[new_img_num] = new_img_2d_3d

        # use the 2d-3d correspondences to find the pose of the new cam
        R_new, C_new = linear_pnp(new_img_2d_3d, K)
        pose_new = np.hstack((R_new, C_new))
        print(pose_new)
        poses[new_img_num] = pose_new

    # plot all the poses
    plot_camera_poses(poses, ref_imgs)

if __name__ == '__main__':
    main()
