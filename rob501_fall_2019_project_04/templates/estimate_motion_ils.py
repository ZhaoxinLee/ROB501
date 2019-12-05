import numpy as np
from numpy.linalg import det, inv, svd
from rpy_from_dcm import rpy_from_dcm
from dcm_from_rpy import dcm_from_rpy
from estimate_motion_ls import estimate_motion_ls

def estimate_motion_ils(Pi, Pf, Si, Sf, iters):
    """
    Estimate motion from 3D correspondences.

    The function estimates the 6-DOF motion of a boby, given a series
    of 3D point correspondences. This method relies on NLS.

    Arrays Pi and Pf store corresponding landmark points before and after
    a change in pose.  Covariance matrices for the points are stored in Si
    and Sf.

    Parameters:
    -----------
    Pi  - 3xn np.array of points (intial - before motion).
    Pf  - 3xn np.array of points (final - after motion).
    Si  - 3x3xn np.array of landmark covariance matrices.
    Sf  - 3x3xn np.array of landmark covariance matrices.

    Outputs:
    --------
    Tfi  - 4x4 np.array, homogeneous transform matrix, frame 'i' to frame 'f'.
    """
    # Initial guess...
    Tfi = estimate_motion_ls(Pi, Pf, Si, Sf)
    C = Tfi[:3, :3]
    t = Tfi[:3, [3]]
    I = np.eye(3)
    rpy = rpy_from_dcm(C).reshape(3, 1)

    Pi = np.vstack((Pi,[1,]*Pi.shape[1]))
    Pf = np.vstack((Pf,[1,]*Pi.shape[1]))

    # Iterate.
    for j in np.arange(iters):
        A = np.zeros((6, 6))
        B = np.zeros((6, 1))

        #--- FILL ME IN ---
        # compute the 4xn array of residuals
        res = Pf - Tfi@Pi

        # the Jacobian matrix
        J = np.zeros((3,6))
        dev_r, dev_p, dev_y = jacob_rpy(rpy)

        # compute the Jacobian for each pair of points
        for i in np.arange(Pi.shape[1]):
            for k in range(3):
                J[k][(k+3)] = 1
                J[k][0] = dev_r[[k],:]@Pi[:3,[i]]
                J[k][1] = dev_p[[k],:]@Pi[:3,[i]]
                J[k][2] = dev_y[[k],:]@Pi[:3,[i]]
            A += J.T@inv(Si[:,:,i])@J
            B += J.T@inv(Si[:,:,i])@res[:3,[i]]

        #------------------

        # Solve system and check stopping criteria if desired...
        theta = inv(A)@B

        # update the parameters and pose matrix
        incr_rpy = theta[0:3].reshape(3, 1)
        rpy = rpy + incr_rpy
        C = dcm_from_rpy(rpy)

        incr_t = theta[3:6].reshape(3, 1)
        t =  t + incr_t

        Tfi = np.vstack((np.hstack((C, t)), np.array([[0, 0, 0, 1]])))

        # iterate until convergence
        if np.linalg.norm(theta) < 1e-20:
            break

    return Tfi

def dcm_jacob_rpy(C):
     # Rotation - convenient!
    cp = np.sqrt(1 - C[2, 0]*C[2, 0])
    cy = C[0, 0]/cp
    sy = C[1, 0]/cp

    dRdr = C@np.array([[ 0,   0,   0],
                       [ 0,   0,  -1],
                       [ 0,   1,   0]])

    dRdp = np.array([[ 0,    0, cy],
                     [ 0,    0, sy],
                     [-cy, -sy,  0]])@C

    dRdy = np.array([[ 0,  -1,  0],
                     [ 1,   0,  0],
                     [ 0,   0,  0]])@C

    return dRdr, dRdp, dRdy

def jacob_rpy(rpy):
    cr = np.cos(rpy[0]).item()
    sr = np.sin(rpy[0]).item()
    cp = np.cos(rpy[1]).item()
    sp = np.sin(rpy[1]).item()
    cy = np.cos(rpy[2]).item()
    sy = np.sin(rpy[2]).item()
    dev_r = np.array([[0, cy*sp*cr + sy*sr, -cy*sp*sr + sy*cr],
                     [0, sy*sp*cr - cy*sr, -sy*sp*sr - cy*cr],
                     [0,            cp*cr,            -cp*sr]])

    dev_p = np.array([[-cy*sp, cy*cp*sr, cy*cp*cr],
                     [-sy*sp, sy*cp*sr, sy*cp*cr],
                     [  -cp,   -sp*sr,   -sp*cr]])

    dev_y = np.array([[-sy*cp, -sy*sp*sr - cy*cr, -sy*sp*cr + cy*sr],
                     [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [  0,            0,            0]])
    return dev_r, dev_p, dev_y
