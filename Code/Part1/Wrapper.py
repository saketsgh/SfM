import random
import os
import cv2
import numpy as np
from GetInliersRANSAC import get_inliers_ransac
from EssentialMatrixFromFundamentalMatrix import estimate_e_matrix
from ExtractCameraPose import extract_camera_pose
from DisambiguateCameraPose import disambiguate_camera_pose
from NonlinearTriangulation import nonlinear_triang
import matplotlib.pyplot as plt


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



def draw_epipolar_lines(F, img_left, img_right, pts_left, pts_right):
    """
    Draw the epipolar lines given the fundamental matrix, left right images
    and left right datapoints
    You do not need to modify anything in this function, although you can if
    you want to.
    :param F: 3 x 3; fundamental matrix
    :param img_left:
    :param img_right:
    :param pts_left: N x 2
    :param pts_right: N x 2
    :return:
    """
    # lines in the RIGHT image
    # corner points
    p_ul = np.asarray([0, 0, 1])
    p_ur = np.asarray([img_right.shape[1], 0, 1])
    p_bl = np.asarray([0, img_right.shape[0], 1])
    p_br = np.asarray([img_right.shape[1], img_right.shape[0], 1])

    # left and right border lines
    l_l = np.cross(p_ul, p_bl)
    l_r = np.cross(p_ur, p_br)

    fig, ax = plt.subplots()
    ax.imshow(img_right)
    ax.autoscale(False)
    ax.scatter(pts_right[:, 0], pts_right[:, 1], marker='o', s=20, c='yellow',
        edgecolors='red')
    for p in pts_left:
        p = np.hstack((p, 1))[:, np.newaxis]
        l_e = np.dot(F, p).squeeze()  # epipolar line
        p_l = np.cross(l_e, l_l)
        p_r = np.cross(l_e, l_r)
        x = [p_l[0]/p_l[2], p_r[0]/p_r[2]]
        y = [p_l[1]/p_l[2], p_r[1]/p_r[2]]
        ax.plot(x, y, linewidth=1, c='blue')

    # lines in the LEFT image
    # corner points
    p_ul = np.asarray([0, 0, 1])
    p_ur = np.asarray([img_left.shape[1], 0, 1])
    p_bl = np.asarray([0, img_left.shape[0], 1])
    p_br = np.asarray([img_left.shape[1], img_left.shape[0], 1])

    # left and right border lines
    l_l = np.cross(p_ul, p_bl)
    l_r = np.cross(p_ur, p_br)

    fig, ax = plt.subplots()
    ax.imshow(img_left)
    ax.autoscale(False)
    ax.scatter(pts_left[:, 0], pts_left[:, 1], marker='o', s=20, c='yellow',
        edgecolors='red')
    for p in pts_right:
        p = np.hstack((p, 1))[:, np.newaxis]
        l_e = np.dot(F.T, p).squeeze()  # epipolar line
        p_l = np.cross(l_e, l_l)
        p_r = np.cross(l_e, l_r)
        x = [p_l[0]/p_l[2], p_r[0]/p_r[2]]
        y = [p_l[1]/p_l[2], p_r[1]/p_r[2]]
        ax.plot(x, y, linewidth=1, c='blue')
    plt.show()

def main():

    path='../../Data/Data/'

    # loading images
    images = load_images(path)

    # defining image pairs
    file_names = ["matches12", "matches13", "matches14", "matches23", "matches24", "matches34", "matches35", "matches36", "matches45"
    , "matches46", "matches56"]
    image_nums = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [2, 4], [2, 5], [3, 4], [3, 5], [4, 5]]

    # given camera calibration matrix
    K = np.array([[568.996140852, 0, 643.21055941],
         [0, 568.988362396, 477.982801038],
         [0, 0, 1]])

    # for each image pairs compute F, E
    i = 0
    for file_name, image_num in zip(file_names, image_nums):

        if(i>0):
            break
        # get inliers and fundamental matrix using RANSAC
        max_inliers_locs, min_outliers_locs, F_max_inliers, pts_left, pts_right = get_inliers_ransac(path, file_name)

        # plotting correspondences for inliers
        plot_correspondences(images[image_num[0]], images[image_num[1]], max_inliers_locs, min_outliers_locs, file_name)

        draw_epipolar_lines(F_max_inliers, images[image_num[0]], images[image_num[1]], pts_left[:100], pts_right[:100])

        # get essential matrix
        E = estimate_e_matrix(F_max_inliers, K)
        C_list, R_list = extract_camera_pose(E)


        # disambiguate camera pose and get the best pose of cam right
        R2, C2, X_list, index = disambiguate_camera_pose(C_list, R_list, K, max_inliers_locs)

        # perform non-linear triangulation to get refined X
        X_list_refined =  nonlinear_triang(R2, C2, X_list, max_inliers_locs, K)


        # compare non-linear triangulation with linear by plot
        linear_vs_non_linear(X_list, X_list_refined, index)

        i+=1
        break








if __name__ == '__main__':
    main()
