import numpy as np
from numpy.linalg import inv, norm
from find_jacobian import find_jacobian
from dcm_from_rpy import dcm_from_rpy
from rpy_from_dcm import rpy_from_dcm

def pose_estimate_nls(K, Twcg, Ipts, Wpts):
    """
    Estimate camera pose from 2D-3D correspondences via NLS.

    The function performs a nonlinear least squares optimization procedure
    to determine the best estimate of the camera pose in the calibration
    target frame, given 2D-3D point correspondences.

    Parameters:
    -----------
    Twcg  - 4x4 homogenous pose matrix, initial guess for camera pose.
    Ipts  - 2xn array of cross-junction points (with subpixel accuracy).
    Wpts  - 3xn array of world points (one-to-one correspondence with Ipts).

    Returns:
    --------
    Twc  - 4x4 np.array, homogenous pose matrix, camera pose in target frame.
    """
    maxIters = 250                          # Set maximum iterations.

    tp = Ipts.shape[1]                      # Num points.
    x_obs = np.reshape(Ipts, (2*tp, 1), 'F')   # Stacked vector of observations.

    res = np.zeros((2*tp, 1))                # Residuals.
    J  = np.zeros((2*tp, 6))                # Jacobian.

    #--- FILL ME IN ---
    R = Twcg[0:3,0:3] # rotational matrix
    T = Twcg[0:3,[3]] # translational matrix
    rpy = rpy_from_dcm(R) # roll, pitch and yaw angle
    param = np.vstack((T,rpy)) # [x,y,z,r,p,y]
    Wpts = np.vstack((Wpts,np.ones(tp)))

    iter = 0
    while True:

        # Compute the predicted saddle points using camera pose guess
        Cpt = np.dot(np.linalg.inv(Twcg),Wpts)
        Cpt = np.delete(Cpt,3,0)
        Cpt = np.dot(K,Cpt)
        x_pred = np.zeros((2,tp))
        for i in range(tp):
            x_pred[0][i] = Cpt[0][i] / Cpt[2][i]
            x_pred[1][i] = Cpt[1][i] / Cpt[2][i]
            J[2*i:2*i+2,:] = find_jacobian(K,Twcg,Wpts[0:3,[i]]) # the Jacobian matrix
        x_pred = np.reshape(x_pred, (2*tp, 1), 'F')

        # Compute the optimal increment using residuals
        res = x_obs - x_pred
        incr = np.dot(np.dot(np.linalg.inv(np.dot(J.T,J)),J.T),res)


        # update the parameters and pose matrix
        param = param+incr
        Twcg[0:3,0:3] = dcm_from_rpy(param[3:6])
        Twcg[0:3,[3]] = param[0:3]

        # iterate until convergence
        if np.linalg.norm(incr) < 1e-12:
            break
        elif iter == maxIters:
            break
        iter += 1

    Twc = Twcg
    #------------------

    return Twc

#----- Functions Go Below -----

def epose_from_hpose(T):
    """Euler pose vector from homogeneous pose matrix."""
    E = np.zeros((6, 1))
    E[0:3] = np.reshape(T[0:3, 3], (3, 1))
    E[3:6] = rpy_from_dcm(T[0:3, 0:3])

    return E

def hpose_from_epose(E):
    """Homogeneous pose matrix from Euler pose vector."""
    T = np.zeros((4, 4))
    T[0:3, 0:3] = dcm_from_rpy(E[3:6])
    T[0:3, 3] = np.reshape(E[0:3], (3,))
    T[3, 3] = 1

    return T
