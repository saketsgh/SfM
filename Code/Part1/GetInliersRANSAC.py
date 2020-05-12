import random
import os
import cv2
import numpy as np
from EstimateFundamentalMatrix import estimate_f_matrix


def get_inliers_ransac(pts_from_txt):

    # seperate points into images
    pts_img1 = pts_from_txt[:, 0:2]
    pts_img2 = pts_from_txt[:, 2:4]

    # conv to homog
    ones = np.ones((pts_img1.shape[0], 1))
    pts_img1 = np.hstack((pts_img1, ones))
    pts_img2 = np.hstack((pts_img2, ones))


    ## RANSAC
    max_inliers = 0

    for i in range(10000):
        # print("iteration number: ",i)

        # condition to check if we get 80% of the inliers.
        # if max_inliers >=0.80*np.shape(pts_from_txt)[0]:
        #     break

        # randomly pick 8 points
        points8 = np.array(random.sample(pts_from_txt, 8), np.float32)

        # estimate fundamental qmatrix
        F = estimate_f_matrix(points8)

        # compute (x2.T)Fx1
        vals = np.abs(np.diag(np.dot(np.dot(pts_img2, F), pts_img1.T)))

        # setting threshold
        # print(vals)
        inliers_index = np.where(vals<0.07)
        outliers_index = np.where(vals>=0.07)

        # inliers_index = np.where(vals<0.05)
        # outliers_index = np.where(vals>=0.05)

        # checking for max_inliersand saving it's index
        if np.shape(inliers_index[0])[0] > max_inliers:
            max_inliers = np.shape(inliers_index[0])[0]
            max_inliers_index = inliers_index
            min_outliers_index = outliers_index
            F_max_inliers = F

    F_max_inliers = estimate_f_matrix(pts_from_txt[max_inliers_index])

    # min_outliers = np.shape(min_outliers_index[0])[0]
    print("max inliers - {}".format(max_inliers))

    return pts_from_txt[max_inliers_index], pts_from_txt[min_outliers_index], F_max_inliers, pts_from_txt[:, 0:2], pts_from_txt[:, 2:4]
