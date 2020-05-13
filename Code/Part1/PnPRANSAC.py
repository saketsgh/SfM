import numpy as np
from LinearPnP import linear_pnp
import random


def compute_reproj_err_all(pts_img_all, M, X_all, ret=False):

    # make X homogenous
    ones = np.ones((X_all.shape[0], 1))
    X_all = np.hstack((X_all, ones))
    X_all = X_all.T

    # get the projected points
    pts_img_proj_all = np.dot(M, X_all)
    pts_img_proj_all = pts_img_proj_all.T
    pts_img_proj_all[:, 0] = pts_img_proj_all[:, 0]/pts_img_proj_all[:, 2]
    pts_img_proj_all[:, 1] = pts_img_proj_all[:, 1]/pts_img_proj_all[:, 2]
    pts_img_proj_all = pts_img_proj_all[:, 0:2]

    # define reprojection error for all points
    reproj_err = pts_img_all - pts_img_proj_all
    reproj_err = reproj_err**2
    reproj_err = np.sum(reproj_err, axis=1)

    if(ret):
        return reproj_err, pts_img_proj_all

    return reproj_err


def pnp_ransac(corresp_2d_3d, K):

    thresh = 20

    # extract point correspondences of given camera
    corresp = corresp_2d_3d

    max_inliers = 0

    # perform RANSAC to estimate the best pose
    for i in range(5000):

        # choose 6 random points and get linear pnp estimate
        corresp6 = np.array(random.sample(corresp, 6), np.float32)
        R, C = linear_pnp(corresp6, K)

        # form the projection matrix
        C = C.reshape((3, 1))
        I = np.identity(3)
        M = np.hstack((I, -C))
        M = np.dot(K, np.dot(R, M))

        # calculate reproj_err for all points
        pts_img_all = corresp[:, 0:2]
        X_all = corresp[:, 2:]

        reproj_err = compute_reproj_err_all(pts_img_all, M, X_all)
        locs = np.where(reproj_err < thresh)[0]
        count = np.shape(locs)[0]
        if count > max_inliers:
            max_inliers = count
            inliers = corresp[locs]
            R_best = R
            C_best = C

    pose_best = np.hstack((R_best, C_best))
    print(max_inliers)
    print("......................................")

    return pose_best, inliers
