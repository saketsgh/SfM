# python code for reading text data, extracting feature points and writing them to txt file for ref

import os
import sys
sys.path.append("../")
import numpy as np

from GetInliersRANSAC import get_inliers_ransac

from utils import MiscFuncs, PlotFuncs

def extract_points(path):

    os.chdir(path)

    for i in range(1, 6):

        file_name = ""
        file_name += "matching" + str(i) + ".txt"
        save_file_name = "matches" + str(i)

        file = open(file_name, 'r')
        content = file.readlines()

        nums = []

        for line in content[1:]:

            nums = line.split()
            num_matches = nums[0]

            matches = nums[6:]
            for j,match in enumerate(matches):

                if(j%3==0):

                    save_file = open(save_file_name + str(match) + ".txt", 'a')

                    # [x1, y1, x2, y2, R, G, B]
                    # Writing to file
                    points = str(nums[4]) + " " + str(nums[5]) + " " + str(matches[j+1]) + " " + str(matches[j+2]) + " " + str(nums[1]) + " " + str(nums[2]) + " " + str(nums[3]) + "\n"
                    save_file.write(points)
                    save_file.close()

        # print(image_ids)


def save_ransac_pts(path):
    image_nums = [[2, 3], [3, 4], [4, 5], [5, 6]]
    corresp_2d_2d = {}
    misc_funcs = MiscFuncs()
    plot_funcs = PlotFuncs()

    cur_path = os.getcwd()
    # thresholds = [0.001, 0.002] perfect for 3 and 4
    thresholds = [0.0025, 0.0025, 0.0025, 0.0025]

    images = misc_funcs.load_images(path)
    for nums, thresh in zip(image_nums, thresholds):
        img_pair = str(nums[0])+str(nums[1])
        file_name = "matches"+img_pair+".txt"
        print("using correspondences from file "+file_name)
        print("performing RANSAC to obtain F matrix\n")
        pts_from_txt = misc_funcs.get_pts_from_txt(path, file_name)
        os.chdir("../../Data/Data")
        pts_from_txt = np.array(pts_from_txt, np.float32)
        inlier_locs, outliers_locs, F, pts_left, pts_right = get_inliers_ransac(pts_from_txt, thresh)
        plot_funcs.plot_img_correspondences(images[nums[0]-1], images[nums[1]-1], inlier_locs, outliers_locs, file_name, save=True)

        save_file_name = "ransac"+img_pair
        for pts in inlier_locs:
            save_file = open(save_file_name + ".txt", 'a')
            save_file.write(str(pts[0])+ " " + str(pts[1]) + " " + str(pts[2]) + " " + str(pts[3]) + "\n")
            save_file.close()
        os.chdir(cur_path)


def main():

    path = "../../../Data/Data/"

    # extract_points(path)
    save_ransac_pts(path)

if __name__ == '__main__':
    main()
