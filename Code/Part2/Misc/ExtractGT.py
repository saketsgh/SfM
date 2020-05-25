# Script of extract the OXTS data , read the timestamps and convert it into ground truth.
# Refereces:
# 1. https://github.com/utiasSTARS/pykitti/tree/master/pykitti
# Change the path accordingly.


import os
from collections import namedtuple
import numpy as np
import linecache
import os
from collections import namedtuple
import numpy as np

OxtsPacket = namedtuple('OxtsPacket',
                        'lat, lon, alt, ' +
                        'roll, pitch, yaw, ' +
                        'vn, ve, vf, vl, vu, ' +
                        'ax, ay, az, af, al, au, ' +
                        'wx, wy, wz, wf, wl, wu, ' +
                        'pos_accuracy, vel_accuracy, ' +
                        'navstat, numsats, ' +
                        'posmode, velmode, orimode')

# Bundle into an easy-to-access structure
OxtsData = namedtuple('OxtsData', 'packet, T_w_imu')


def subselect_files(files, indices):
    try:
        files = [files[i] for i in indices]
    except:
        pass
    return files


def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def load_oxts_packets_and_poses(oxts_files,time_files):
    """Generator to read OXTS ground truth data.
       Poses are given in an East-North-Up coordinate system 
       whose origin is the first GPS position.
    """
    # Scale for Mercator projection (from first lat value)
    
    # Origin of the global coordinate system (first GPS position)
    origin = None
    originR = None 

    oxts = []
    Z = []
    for filename,timestamp in zip(oxts_files,time_files):
        with open(filename[:-1], 'r') as f:
            for line in f.readlines():
                
                line = line.split()
                # Last five entries are flags and counts
                line[:-5] = [float(x) for x in line[:-5]]
                line[-5:] = [int(float(x)) for x in line[-5:]]
		
                packet = OxtsPacket(*line)
                scale = np.cos(packet.lat * np.pi / 180.)
                print(scale)
                R, t = pose_from_oxts_packet(packet, scale)                	
                q = rot2Quat(R)
                
        	
                if origin is None:
                    origin = t
                    originR = q
        		    

                T_w_imu = transform_from_rot_trans(R, t - origin)
                Z.append([timestamp[17:-5],"{:.5f}".format(T_w_imu[:,3][0]),"{:.5f}".format(T_w_imu[:,3][1]),"{:.5f}".format(T_w_imu[:,3][2]),"{:.5f}".format(q[0]-originR[0]),"{:.5f}".format(q[1]-originR[1]),"{:.5f}".format(q[2]-originR[2]),"{:.5f}".format(q[3]-originR[3]+1)])
                oxts.append(OxtsData(packet, T_w_imu))
    return oxts, Z

def rot2Quat(rot):

	qxx,qyx,qzx,qxy,qyy,qzy,qxz,qyz,qzz = rot.flatten()
	m = np.array([[qxx-qyy-qzz,0, 0, 0],[qyx+qxy,qyy-qxx-qzz,0,0],
		[qzx+qxz,qzy+qyz,qzz-qxx-qyy,0],[qyz-qzy,qzx-qxz,qxy-qyx,qxx+qyy+qzz]])/3.0
	val,vec = np.linalg.eigh(m)
	q = vec[[3,0,1,2],np.argmax(val)]
	if q[0]<0:
		q = -q

	return q

def pose_from_oxts_packet(packet, scale):
    """Helper method to compute a SE(3) pose matrix from an OXTS packet.
    """
    er = 6378137.  # earth radius (approx.) in meters

    # Use a Mercator projection to get the translation vector
    tx = scale * packet.lon * np.pi * er / 180.
    ty = scale * er * \
        np.log(np.tan((90. + packet.lat) * np.pi / 360.))
    tz = packet.alt
    t = np.array([tx, ty, tz])

    # Use the Euler angles to get the rotation matrix
    Rx = rotx(packet.roll)
    Ry = roty(packet.pitch)
    Rz = rotz(packet.yaw)
    R = Rz.dot(Ry.dot(Rx))

    # Combine the translation and rotation into a homogeneous transform
    
    return R, t.reshape(3,-1)

def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)

    return np.vstack((np.hstack([R, t])))



def main():
    n = len(linecache.getlines('/media/patodichayan/DATA/733/PFinal/Test/2011_09_26/gt.txt')[0:-1])
    
    for i in range(n-1):
        filename = linecache.getline('/media/patodichayan/DATA/733/PFinal/Test/2011_09_26/gt.txt',i+1)
        test_files = linecache.getlines('/media/patodichayan/DATA/733/PFinal/Test/2011_09_26/gt.txt')[i:i+3]
        time_files = linecache.getlines('/media/patodichayan/DATA/733/PFinal/Test/2011_09_26/2011_09_26_drive_0015_sync/oxts/timestamps.txt')[i:i+3]
        result,Z = load_oxts_packets_and_poses(test_files,time_files)
        np.savetxt('/media/patodichayan/DATA/733/PFinal/Test/2011_09_26/2011_09_26_drive_0015_sync/gt3/{}.txt'.format(filename[93:-5]),(np.array(Z)).reshape(3,-1),fmt='%s')
        	

main()		
	

