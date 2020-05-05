import numpy as np 


def estFundMat(pts1, pts2):
	'''This returns the fundamental matrix mapping points as 
		pts2^T F pts1 = 0.'''

	pts1 = np.hstack((pts1,np.ones((pts1.shape[0],1))))
	pts2 = np.hstack((pts2,np.ones((pts2.shape[0],1))))
	
	A = np.vstack(([pts1[:,0]*pts2[:,0]],[pts1[:,0]*pts2[:,1]],[pts1[:,0]*pts2[:,2]],
			[pts1[:,1]*pts2[:,0]],[pts1[:,1]*pts2[:,1]],[pts1[:,1]*pts2[:,2]],
			[pts1[:,2]*pts2[:,0]],[pts1[:,2]*pts2[:,1]],[pts1[:,2]*pts2[:,2]])).T
	
	U,S,V = np.linalg.svd(A)
	x = V.T[:,-1]
	F = x.reshape((3,3)).T
	
	# Make F become rank 2 explicitly.
	U, S, VT = np.linalg.svd(F)
	S[2] = 0.0
	F = U.dot(np.diag(S)).dot(VT)

	return F
