# Project- Learning SfM

### Authors: 
1. Chayan Kumar Patodi ([ckp1804@terpmail.umd.edu](mailto:ckp1804@terpmail.umd.edu), UID: 116327428)
2. Saket Seshadri Gudimetla Hanumath (saketsgh@terpmail.umd.edu, UID: 116332293)

### To Run:
1. Phase 1:
To run the program, type the following:-"  python2.7 Wrapper.py  ".
Every other file has default arguments and should work with the existing directory structure.

Note: More text files containing points were generated for convinence. They can be found inside the data folder inside the data folder.

2. Phase 2:
train.py : file with the changes and was used to train the model.  Just give the correct path to the flag : dataset_dir .

	test_kitti_depth.py : change the directory path to the checkpoint folder.
test_kitti_pose.py :  change the directory path to the checkpoint folder.

	Inside the evaluation folder, you can find the depth and pose evaluation files. Give the path to the predicted files. 

	Inside the Misc folder, you find the additional scripts that were created to extract the ground truth and restructure the dataset.





The testing data (2011_09_26_drive_0001) that we used was downloaded from here :
https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip

##### *This was also used for pose estimation ,instead of the odometry set. We just resturctued the data accordingly.*

The trained models can be found on : https://umd.box.com/s/qrz851xzn35uwjcpgp07gc9c9b5ywg44
