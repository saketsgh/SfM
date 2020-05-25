
# Saving the file names of the files of OXTS data, so as to use further.
# Read the oxts file name and write them to a text file.
# Make changes to the path.

import os
from glob import glob

with open("oxtsfiles.txt", "w") as a:
	for files in sorted(glob(r'/media/patodichayan/DATA/733/PFinal/Test/2011_09_26/2011_09_26_drive_0015_sync/oxts/data/*.txt')):
			a.write(files + os.linesep) 
