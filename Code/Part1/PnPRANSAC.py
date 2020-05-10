import numpy as np
from LinearPnP import linear_pnp
import random

def compute_reproj_err_all(pts_img_all, M, X_all):

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

    return reproj_err



def pnp_ransac(corresp_2d_3d, K):

    # for each set of correspondences for images 3-6
    pose_new = {}
    for p in corresp_2d_3d:

        if(p<3):
            continue

        # extract point correspondences of given camera
        corresp = corresp_2d_3d[p]

        max_inliers = 0

        # perform RANSAC to estimate the best pose
        for i in range(2000):

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
            locs = np.where(reproj_err < 30)
            count = np.shape(locs)[0]

            if count > max_inliers:
                max_inliers = count
                inlier_locs = locs
                R_best = R
                C_best = C

        inlier_err = reproj_err[inlier_locs]
        print(inlier_err)
        pose_best = np.hstack((R_best, C_best))
        pose_new[p] = pose_best
        print(max_inliers)
        print(np.mean(inlier_err))
        print("......................................")

    return pose_new
