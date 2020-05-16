import numpy as np
from scipy.optimize import least_squares
from Misc.utils import MiscFuncs

def rot2Quat(rot):

	qxx,qyx,qzx,qxy,qyy,qzy,qxz,qyz,qzz = rot.flatten()
	m = np.array([[qxx-qyy-qzz,0, 0, 0],[qyx+qxy,qyy-qxx-qzz,0,0],
		[qzx+qxz,qzy+qyz,qzz-qxx-qyy,0],[qyz-qzy,qzx-qxz,qxy-qyx,qxx+qyy+qzz]])/3.0
	val,vec = np.linalg.eigh(m)
	q = vec[[3,0,1,2],np.argmax(val)]
	if q[0]<0:
		q = -q

	return q


def quat2Rot(q):

	w,x,y,z = q
	Nq = w*w+x*x+y*y+z*z
	if Nq < np.finfo(np.float).eps:
		return np.eye(3)
	s = 2.0/Nq
	X = x*s
	Y = y*s
	Z = z*s
	wX = w*X; wY = w*Y; wZ = w*Z
	xX = x*X; xY = x*Y; xZ = x*Z
	yY = y*Y; yZ = y*Z; zZ = z*Z
	rot =  np.array([[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])

	return rot


def compute_reproj_err(M, pt_img, X):

    # make X homogenous
    X = X.reshape((3, 1))
    X = np.append(X, 1)

    # get the projected image points
    pt_img_proj = np.dot(M, X)

    # convert to non homog form
    pt_img_proj[0] = pt_img_proj[0]/pt_img_proj[2]
    pt_img_proj[1] = pt_img_proj[1]/pt_img_proj[2]

    reproj_err = ((pt_img[0] - pt_img_proj[0])**2) + ((pt_img[1] - pt_img_proj[1])**2)

    return reproj_err


def optimize_params(x0, K, pts_img_all, X_all):

	# calculate reprojection error
	reproj_err_all = []
	R = quat2Rot(x0[:4])
	C = x0[4:]

	misc_funcs = MiscFuncs()
	M = misc_funcs.get_projection_matrix(K, R, C)

	for pt_img, X in zip(pts_img_all, X_all):
	    reproj_err = compute_reproj_err(M, pt_img, X)
	    reproj_err_all.append(reproj_err)
	reproj_err_all = np.array(reproj_err_all)
	reproj_err_all = reproj_err_all.reshape(reproj_err_all.shape[0],)

	return reproj_err_all


def nonlinear_pnp(K, pose, corresp_2d_3d):

	# extract image points
	poses_non_linear = {}

	# extract point correspondences of given camera
	corresp = corresp_2d_3d
	pts_img_all = corresp[:, 0:2]
	X_all = corresp[:, 2:]

	# make the projection projection matrix
	R = pose[:, 0:3]
	C = pose[:, 3]
	C = C.reshape((3, 1))

	# convert rotation matrix to quaternion form
	Q = rot2Quat(R)

	# defining the paramerter to optimize
	x0 = np.append(Q, C)

	result = least_squares(fun=optimize_params, x0=x0, args=(K, pts_img_all, X_all), ftol=1e-10)
	opt = result.x

	# quaternion to rotation matrix
	R_best = quat2Rot(opt[:4])
	C_best = opt[4:]
	C_best = C_best.reshape((3, 1))
	pose_best = np.hstack((R_best, C_best))

	return pose_best
