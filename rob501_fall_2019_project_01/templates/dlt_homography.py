import numpy as np
from numpy.linalg import inv, norm
from scipy.linalg import null_space

def dlt_homography(I1pts, I2pts):
    """
    Find perspective Homography between two images.

    Given 4 points from 2 separate images, compute the perspective homography
    (warp) between these points using the DLT algorithm.

    Parameters:
    -----------
    I1pts  - 2x4 np.array of points from Image 1 (each column is x, y).
    I2pts  - 2x4 np.array of points from Image 2 (in 1-to-1 correspondence).

    Returns:
    --------
    H  - 3x3 np.array of perspective homography (matrix map) between image coordinates.
    A  - 8x9 np.array of DLT matrix used to determine homography.
    """
    #--- FILL ME IN ---
    x = I1pts[0]
    y = I1pts[1]
    u = I2pts[0]
    v = I2pts[1]

    # concatenate the matrix A using 4 points correspondences
    A = np.empty([0,9], int)
    for i in range(4):
        A = np.append(A,[[-x[i],-y[i],-1,0,0,0,u[i]*x[i],u[i]*y[i],u[i]],[0,0,0,-x[i],-y[i],-1,v[i]*x[i],v[i]*y[i],v[i]]],axis=0)

    # calculate H using null space
    H = null_space(A)
    H = H/H[8]
    H = np.reshape(H,(3,3))
    #------------------

    return H, A
