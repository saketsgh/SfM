"""
CMSC733 Spring 2020: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 3: Structure From Motion

@file    Wrapper.py
@author  Saket Seshadri Gudimetla Hanumath
@author  Chayan Kumar Patodi
"""
import random
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from GetInliersRANSAC import get_inliers_ransac
from EssentialMatrixFromFundamentalMatrix import estimate_e_matrix
from ExtractCameraPose import extract_camera_pose
from DisambiguateCameraPose import disambiguate_camera_pose
from NonlinearTriangulation import nonlinear_triang
from LinearPnP import linear_pnp
from PnPRANSAC import pnp_ransac, compute_reproj_err_all
from NonlinearPnP import nonlinear_pnp
from Misc.utils import PlotFuncs
from Misc.utils import MiscFuncs


def main():

    # path to load the data from
    path='../../Data/Data/'

    # loading images
    misc_funcs = MiscFuncs()
    images = misc_funcs.load_images(path)
    print("loaded images\n")
    # names of file_names to load correspondences from
    # "matches12", "matches13","matches14", "matches23",
    # "matches24", "matches34", "matches35", "matches36",
    # "matches45", "matches46", "matches56"
    # file_nums = [[1, 2], [1, 3], [1, 4], [2, 3],[2, 4], [3, 4],[3, 5], [3, 6],[4, 5], [4, 6],[5, 6]]
    #
    # for nums in file_nums:
    #     img_l = nums[0]
    #     img_r = nums[1]
    #     file_name = "matches"+str(img_l)+str(img_r)+".txt"
    #     pts_from_txt = misc_funcs.get_pts_from_txt(path, file_name)
    #     pts_from_txt = np.array(pts_from_txt, np.float32)
    #     max_inliers_locs, min_outliers_locs, F_max_inliers, pts_left, pts_right = get_inliers_ransac(pts_from_txt)
    #
    #     # plotting correspondences for inliers
    #     plot_funcs = PlotFuncs()
    #     print("plotting correspondences between images - " + str(nums[0]) + str(nums[1]))
    #     plot_funcs.plot_img_correspondences(images[nums[0]-1], images[nums[1]-1], max_inliers_locs, min_outliers_locs, file_name)

    # given camera calibration matrix
    K = np.array([
                [568.996140852, 0, 643.21055941],
                [0, 568.988362396, 477.982801038],
                [0, 0, 1]
                ])

    # define camera 1 as the world pose
    M1 = np.identity(4)
    M1 = M1[0:3, :]
    M1 = np.dot(K, M1)

    # for each image pairs compute F, E
    '''.............................get inliers and fundamental matrix using RANSAC...........................'''
    # load correspondences between image 1 and 2
    file_name = "matches12.txt"
    print("using correspondences from file "+file_name)
    print("performing RANSAC to obtain F matrix\n")
    pts_from_txt = misc_funcs.get_pts_from_txt(path, file_name)
    pts_from_txt = np.array(pts_from_txt, np.float32)
    max_inliers_locs, min_outliers_locs, F_max_inliers, pts_left, pts_right = get_inliers_ransac(pts_from_txt)


    # plotting correspondences for inliers
    plot_funcs = PlotFuncs()
    print("plotting correspondences between images 1 and 2\n")
    plot_funcs.plot_img_correspondences(images[0], images[1], max_inliers_locs, min_outliers_locs, file_name)

    '''.............................essential matrix...........................'''
    print("estimating E matrx\n")
    E = estimate_e_matrix(F_max_inliers, K)
    C2_list, R2_list = extract_camera_pose(E)


    '''.............................disambiguate pose of cam 2...........................'''
    print("disambiguating image 2's camera pose")
    R2, C2, X_list, index = disambiguate_camera_pose(M1, C2_list, R2_list, K, max_inliers_locs)

    # construct projection matrix of camera 2
    I = np.identity(3)
    M2 = np.hstack((I, -C2))
    M2 = np.dot(K, np.dot(R2, M2))

    '''.............................non linear triangulation...........................'''
    print("performing non-linear triangulation to refine X\n")
    X_list_refined =  nonlinear_triang(M1, M2, X_list, max_inliers_locs, K)

    # compare non-linear triangulation with linear by plot
    print("comapring linear and non linear triangulation\n")
    plot_funcs.linear_vs_non_linear(X_list, X_list_refined, index)

    # create a dict to store all the poses
    poses = {}

    # store the pose of 1st and 2nd cam
    poses[1] = np.identity(4)[0:3, :]
    C2 = C2.reshape((3, 1))
    poses[2] = np.hstack((R2, C2))

    '''.............................Perspective-n-Points...........................'''
    print("performing linear PnP to estimate pose of cameras 3-6")
    # using correspondences between the following image pairs for PnP
    image_nums = [[2, 3], [2, 4], [3, 5], [5, 6]]
    print("using the image pairs --> {}\n".format(image_nums))

    # create a dict consisting of 2d-3d correspondences of all images
    corresp_2d_3d = {}

    # first we need to get inliers of image i(3-6) wrt previously estimated camera pose so that we
    # match the 2D image point with the already calculated 3D point
    img1_2d_3d = max_inliers_locs[:, 0:2]
    X_list_refined = np.reshape(X_list_refined, (img1_2d_3d.shape[0], 3))
    img1_2d_3d = np.hstack((img1_2d_3d, X_list_refined))
    corresp_2d_3d[1] = img1_2d_3d

    # same thing for image 2
    img2_2d_3d = max_inliers_locs[:, 2:4]
    img2_2d_3d = np.hstack((img2_2d_3d, X_list_refined))
    corresp_2d_3d[2] = img2_2d_3d

    # estimate pose for the remaining cams
    for _, nums in enumerate(image_nums):

        ref_img_num = nums[0]
        new_img_num = nums[1]

        file_name = "matches"+str(ref_img_num)+str(new_img_num)+".txt"
        print("using correspondences from file " + file_name)

        # get the 2d-3d correspondences for the ref image
        ref_img_2d_3d = corresp_2d_3d[ref_img_num]

        # next we must compare it with the points found using given matches
        matches = misc_funcs.get_pts_from_txt(path, file_name)
        matches = np.array(matches, np.float32)
        # print("p --> {}".format(nums[1]))

        new_img_2d_3d = misc_funcs.get_2d_corresp(ref_img_2d_3d, matches)
        print("shape of 2d-3d correspondences {}".format(np.shape(new_img_2d_3d)))

        # add it to ref_imgs for future use
        corresp_2d_3d[new_img_num] = new_img_2d_3d

        # use the 2d-3d correspondences to find the pose of the new cam
        R_new, C_new = linear_pnp(new_img_2d_3d, K)
        C_new = C_new.reshape((3, 1))
        poses[new_img_num] = np.hstack((R_new, C_new))

        # plot_funcs.plot_reproj_points(images[nums[0]-1], nums[0], np.float32(ref_img_2d_3d[:, 0:2]), np.float32(matches[:, 0:2]))


    # plot all the poses
    print("plotting all the camera poses and their respective correspondences\n")
    plot_funcs.plot_camera_poses(poses, corresp_2d_3d)

    '''.............................PnP RANSAC...........................'''
    print("performing PnP RANSAC to refine the poses")
    poses_pnp_ransac = pnp_ransac(corresp_2d_3d, K)

    '''.............................Non-linear PnP...........................'''
    print("performing Non-linear PnP to obtain optimal pose")
    poses_non_linear = nonlinear_pnp(K, poses_pnp_ransac, corresp_2d_3d)

    # print reprojection error after non-linear pnp
    mean_proj_error = {}
    for p in poses_non_linear:

        pts_img_all = corresp_2d_3d[p][:, 0:2]
        X_all = corresp_2d_3d[p][:, 2:]
        R = poses_non_linear[p][:, 0:3]
        C = poses_non_linear[p][:, 3].reshape((3, 1))

        # compute projection matrix
        M = np.dot(K, np.dot(R, np.hstack((np.identity(3), -C))))
        reproj_errors, pts_img_reproj_all = compute_reproj_err_all(pts_img_all, M, X_all, ret=True)

        mean_proj_error[p] = np.mean(reproj_errors)

        # plotting reprojected points
        plot_funcs.plot_reproj_points(images[p-1], p, np.float32(pts_img_all), np.float32(pts_img_reproj_all))
        print(np.mean(reproj_errors))

    print(mean_proj_error)

if __name__ == '__main__':
    main()
