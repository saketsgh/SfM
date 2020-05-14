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
from LinearTriangulation import linear_triagulation
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
    # #
    # for nums in file_nums:
    #     img_l = nums[0]
    #     img_r = nums[1]
    #     file_name = "matches"+str(img_l)+str(img_r)+".txt"
    #     pts_from_txt = misc_funcs.get_pts_from_txt(path, file_name)
    #     pts_from_txt = np.array(pts_from_txt, np.float32)
    #     # max_inliers_locs, min_outliers_locs, F_max_inliers, pts_left, pts_right = get_inliers_ransac(pts_from_txt)
    #
    #     # plotting correspondences for inliers
    #     plot_funcs = PlotFuncs()
    #     print("plotting correspondences between images - " + str(nums[0]) + str(nums[1]))
    #     print("count - {}".format(pts_from_txt.shape[0]))
    #     plot_funcs.plot_img_correspondences(images[nums[0]-1], images[nums[1]-1], pts_from_txt, [], file_name)
    np.random.seed(2)
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
    pose_set = {}
    mean_proj_error = {}

    # store the pose of 1st and 2nd cam
    pose_set[1] = np.identity(4)[0:3, :]
    C2 = C2.reshape((3, 1))
    pose_set[2] = np.hstack((R2, C2))

    '''.............................Perspective-n-Points...........................'''
    print("performing linear PnP to estimate pose of cameras 3-6")
    # using correspondences between the following image pairs for PnP
    image_nums = [[2, 3], [3, 4], [4, 5], [5, 6]]
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

    # err_all = []
    # for p in img2_2d_3d:
    #
    #     x, y = p[0], p[1]
    #     X = p[2:].reshape((3, 1))
    #     X = np.append(X, 1)
    #
    #     proj = np.dot(M2, X)
    #     x_p = proj[0]/proj[2]
    #     y_p = proj[1]/proj[2]
    #
    #     err = (x-x_p)**2 + (y-y_p)**2
    #     err_all.append(err)
    #
    # err_all = np.array(err_all)
    # print(np.mean(err_all))

    # estimate pose for the remaining cams
    for _, nums in enumerate(image_nums):

        ref_img_num = nums[0]
        new_img_num = nums[1]
        img_pair = str(nums[0])+str(nums[1])
        file_name = "ransac"+img_pair+".txt"
        print("using correspondences from file " + file_name)

        # get the 2d-3d correspondences for the 1st ref image
        ref_img_2d_3d = corresp_2d_3d[ref_img_num]

        # next we must compare it with the points found using given matches
        matches_2d_2d = misc_funcs.get_ransac_pts_from_txt(path, file_name)
        matches_2d_2d = np.array(matches_2d_2d)

        # obtain the 3D corresp for the new image
        new_img_2d_3d, remaining_2d_2d = misc_funcs.get_2d_3d_corresp(ref_img_2d_3d, matches_2d_2d)
        print("shape of 2d-3d correspondences {}".format(np.shape(new_img_2d_3d)))

        '''.............................PnP RANSAC...........................'''
        print("performing PnP RANSAC to refine the poses")
        pose_pnp_ransac, pnp_inlier_corresp = pnp_ransac(new_img_2d_3d, K)

        '''.............................Non-linear PnP...........................'''
        print("performing Non-linear PnP to obtain optimal pose")
        pose_non_linear = nonlinear_pnp(K, pose_pnp_ransac, pnp_inlier_corresp)

        R_new = pose_non_linear[:, 0:3]
        C_new = pose_non_linear[:, 3].reshape((3, 1))
        M_new = misc_funcs.get_projection_matrix(K, R_new, C_new)

        # construct projection matrix of ref image
        R_ref = pose_set[nums[0]][:, 0:3]
        C_ref = pose_set[nums[0]][:, 3].reshape((3, 1))
        M_ref = misc_funcs.get_projection_matrix(K, R_ref, C_ref)

        # find the 2d-3d mapping for the remaining image points in the new image by doing triangulation
        X_new = linear_triagulation(M_ref, C_new, R_new, K, remaining_2d_2d)
        X_new_ref = nonlinear_triang(M_ref, M_new, X_new, remaining_2d_2d, K)
        X_new_ref = X_new_ref.reshape((remaining_2d_2d.shape[0], 3))

        remaining_2d_3d = remaining_2d_2d[:, 2:4]
        print("points before adding remaining corresp - {}".format(new_img_2d_3d.shape))
        new_img_2d_3d = np.vstack((new_img_2d_3d, np.hstack((remaining_2d_3d, X_new_ref))))
        print("points after adding remaining corresp - {}".format(new_img_2d_3d.shape))
        print("......................................")

        # Save the correspondences (2D-3D) and the poses
        corresp_2d_3d[new_img_num] = new_img_2d_3d
        pose_set[new_img_num] = np.hstack((R_new, C_new))

        # use the 2d-3d correspondences to find the pose of the new cam
        # R_new, C_new = linear_pnp(new_img_2d_3d, K)
        # C_new = C_new.reshape((3, 1))
        # poses[new_img_num] = np.hstack((R_new, C_new))

        # plot all the poses
        # print("plotting all the camera poses and their respective correspondences\n")
        # plot_funcs.plot_camera_poses(poses, corresp_2d_3d)

        # print reprojection error after non-linear pnp
        pts_img_all = new_img_2d_3d[:, 0:2]
        X_all = new_img_2d_3d[:, 2:]

        reproj_errors, pts_img_reproj_all = compute_reproj_err_all(pts_img_all, M_new, X_all, ret=True)

        mean_proj_error[new_img_num] = np.mean(reproj_errors)

        # plotting reprojected points
        plot_funcs.plot_reproj_points(images[new_img_num-1], new_img_num, np.float32(pts_img_all), np.float32(pts_img_reproj_all), save=True)

    print(mean_proj_error)
    print(pose_set)



if __name__ == '__main__':
    main()
