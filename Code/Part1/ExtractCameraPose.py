import numpy as np



def check_determinant(C,R):
    if np.linalg.det(R) < 0:
        return -C, -R

    else:
        return C, R



def extract_camera_pose(E):

    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    U,_,Vt = np.linalg.svd(E)

    # Possible Configurations:

    C1 = U[:, 2]
    R1 = (np.dot(U, np.dot(W, Vt)))
    C2 = -U[:, 2]
    R2 = (np.dot(U, np.dot(W, Vt)))
    C3 = U[:, 2]
    R3 = (np.dot(U, np.dot(W.T, Vt)))
    C4 = -U[:, 2]
    R4 = (np.dot(U, np.dot(W.T, Vt)))

    C1,R1 = check_determinant(C1,R1)
    C2,R2 = check_determinant(C2,R2)
    C3,R3 = check_determinant(C3,R3)
    C4,R4 = check_determinant(C4,R4)


    C = np.array([C1, C2, C3, C4])
    R = np.array([R1, R2, R3, R4])

    return C, R
