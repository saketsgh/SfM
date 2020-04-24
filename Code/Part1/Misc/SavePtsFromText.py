# python code for reading text data, extracting feature points and writing them to txt file for ref

import os

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


def main():

    path = "../../../Data/Data/"

    extract_points(path)


if __name__ == '__main__':
    main()
