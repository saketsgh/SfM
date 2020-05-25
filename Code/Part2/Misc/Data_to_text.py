# Script to write the path of the files of the test dataset to a text file. 
# This file would be used to extract the file names.
# Change the path.


import os

with open("testfiles.txt", "w") as a:
	for path, subdirs, files in os.walk(r'/media/patodichayan/DATA/733/PFinal/Test/2011_09_26/'):
		for filename in files:

			if filename.endswith(".png"):
				f = os.path.join(path,filename)
]				a.write(str((f[41:51] + "/" + f[41:])) + os.linesep) 
