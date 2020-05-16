import os
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl


class PlotFuncs:

    def __init__(self,):
        self.misc_funcs = MiscFuncs()


    def linear_vs_non_linear(self, X_linear, X_non_linear, index):

        # extract the x and the z components

        X_linear = np.array(X_linear)
        X_linear = X_linear.reshape((X_linear.shape[0], -1))

        x_l = X_linear[:, 0]
        z_l = X_linear[:, 2]

        # x_indices = range(0, np.shape(X_non_linear)[0], 3)
        # z_indices = range(2, np.shape(X_non_linear)[0], 3)
        # x_nl = X_non_linear[x_indices]
        # z_nl = X_non_linear[z_indices]
        x_nl = X_non_linear[:, 0]
        z_nl = X_non_linear[:, 2]

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


    def plot_img_correspondences(self, imgA, imgB, inlier_locs, outlier_locs, matchesImg="matches12", save=False):

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


    def plot_triangulated_points(self, X_list, i, C, R):

        # plot the points
        X_list = np.array(X_list)

        # reshape
        X_list = X_list.reshape((X_list.shape[0], 3))

        # extract x and z
        x = X_list[:, 0]
        z = X_list[:, 2]

        colormap = np.array(['y', 'b', 'c', 'r'])

        ax = plt.gca()

        # extract euler angles(w.r.t y since plot is x vs z) to plot markers
        ax.plot(0, 0, marker=mpl.markers.CARETDOWN, markersize=15, color = 'k')

        euler_angles = self.misc_funcs.rotationMatrixToEulerAngles(R)
        angles_camera = np.rad2deg(euler_angles)

        # plot the cameras
        t = mpl.markers.MarkerStyle(marker=mpl.markers.CARETDOWN)
        t._transform = t.get_transform().rotate_deg(int(angles_camera[1]))

        ax.scatter((C[0]), (C[2]), marker=t, s=250, color=colormap[i])
        ax.scatter(x, z, s=4, color=colormap[i])


    def plot_camera_poses(self, poses, corresp_2d_3d, save=False):

        colormap = ['y', 'b', 'c', 'm', 'r', 'k']

        fig = plt.figure()
        ax = fig.add_subplot(111)


        for c, p in zip(colormap, poses):

            # extract the camera pose
            pose = poses[p]
            R = pose[:, 0:3]
            C = pose[:, 3]

            # extract the 3d correspondences
            corr2d3d = corresp_2d_3d[p]
            X = corr2d3d[:, 2:]

            # plot the cameras
            euler_angles = self.misc_funcs.rotationMatrixToEulerAngles(R)
            angles_camera = np.rad2deg(euler_angles)

            t = mpl.markers.MarkerStyle(marker=mpl.markers.CARETDOWN)
            t._transform = t.get_transform().rotate_deg(int(angles_camera[1]))
            ax = plt.gca()
            ax.scatter((C[0]), (C[2]), marker=t, s=250, color=c)
            ax.scatter(X[:, 0], X[:, 2], s=4, color=c)
        plt.xlim(-30, 20)
        plt.ylim(-30, 40)
        if(save):
            plt.savefig('op.png')
        plt.show()




    def plot_reproj_points(self, image, image_num, pts_img_all, pts_img_reproj_all, save=False):

        for p, p_rep in zip(pts_img_all, pts_img_reproj_all):
            cv2.circle(image, (p[0], p[1]), 3, (255, 0, 0), 1)
            cv2.circle(image, (p_rep[0], p_rep[1]), 3, (0, 0, 255), 1)
            cv2.line(image, (p[0], p[1]), (p_rep[0], p_rep[1]), (0, 255, 255), 1)

        image_name = "reproj_pts_"+str(image_num)
        cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(image_name, (image.shape[0]/2, image.shape[0]/2))
        cv2.imshow(image_name, image)
        cv2.waitKey(0)
        if(save):
            cv2.imwrite(image_name+".jpg", image)
        cv2.destroyAllWindows()



class MiscFuncs:

    def get_pts_from_txt(self, path, file_name):

        os.chdir(path)

        file = open(file_name, 'r')
        content = file.readlines()

        pts_from_txt = []

        for line in content:
            x1, y1, x2, y2, r, g, b = line.split()

            pts_from_txt.append([np.float32(x1), np.float32(y1), np.float32(x2), np.float32(y2), int(r), int(g), int(b)])

        os.chdir('../../Code/Part1')
        return pts_from_txt


    def get_ransac_pts_from_txt(self, path, file_name):

        os.chdir(path)

        file = open(file_name, 'r')
        content = file.readlines()

        pts_from_txt = []

        for line in content:
            x1, y1, x2, y2 = line.split()

            pts_from_txt.append([np.float32(x1), np.float32(y1), np.float32(x2), np.float32(y2)])

        os.chdir('../../Code/Part1')
        return pts_from_txt


    # Checks if a matrix is a valid rotation matrix.
    def isRotationMatrix(self, R):

        Rt = np.transpose(R)

        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6


    def rotationMatrixToEulerAngles(self, R):

        # check if rot mat is correct
        assert (self.isRotationMatrix(R))

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


    def load_images(self, path):

        images = []

        for i in range(1, 7):
            img = cv2.imread(path+str(i)+".jpg")
            images.append(img)

        return images


    def get_2d_3d_corresp(self, ref_img_2d_3d, matches):

        new_img_2d_3d = []
        remaining_2d_2d = []

        for pts in matches:
            matches_ref_img = pts[0:2]
            matches_new_img = pts[2:4]

            # compare with all other points in ref image to get
            ref_2d = ref_img_2d_3d[:, 0:2]
            ssd = ref_2d - matches_ref_img
            ssd = ssd**2
            ssd = np.sum(ssd, axis=1)
            ssd = np.sqrt(ssd)
            locs = np.where(ssd < 1e-3)[0]

            if(np.shape(locs)[0] == 1):
                # we got the same point from inliers calculated in previous steps and from the matches
                new_img_2d_3d.append([matches_new_img[0], matches_new_img[1],
                ref_img_2d_3d[locs][0][2], ref_img_2d_3d[locs][0][3], ref_img_2d_3d[locs][0][4]])

            else:
                remaining_2d_2d.append(pts)

        return np.array(new_img_2d_3d), np.array(remaining_2d_2d)


    def get_projection_matrix(self, K, R, C):

        R = R.reshape((3, 3))
        C = C.reshape((3, 1))
        I = np.identity(3)
        M = np.hstack((I, -C))
        M = np.dot(K, np.dot(R, M))

        return M
