import numpy as np

def find_jacobian(K, Twc, Wpt):
    """
    Determine the Jacobian for NLS camera pose optimization.

    The function computes the Jacobian of an image plane point with respect
    to the current camera pose estimate, given a landmark point. The
    projection model is the simple pinhole model.

    Parameters:
    -----------
    K    - 3x3 np.array, camera intrinsic calibration matrix.
    Twc  - 4x4 np.array, homogeneous pose matrix, current guess for camera pose.
    Wpt  - 3x1 world point on calibration target (one of n).

    Returns:
    --------
    J  - 2x6 np.array, Jacobian matrix (columns are tx, ty, tz, r, p, q).
    """
    #--- FILL ME IN ---

    # compute the coordinates of world point relative to the camera
    R = Twc[0:3,0:3]
    T = Twc[0:3,3]
    Wpt_r = Wpt.T - T # before translational motion
    Cpt = np.dot(R.T,Wpt_r.T) # the point coordinates in camera frame, R is an orthogonal matrix

    # compute the roll, pitch and yaw angle
    rpy = rpy_from_dcm(R)

    # take partial derivatives for r, p and y
    dev_r, dev_p, dev_y = jacob_rpy(rpy)

    # compute Jacobian matrix with respect to x, y, z, r, p and y
    # use quotient rule
    J = np.zeros((2,6))
    for i in range(2):
        J[i][0] = -K[i][i] * (R[0][i] * Cpt[2] - Cpt[i] * R[0][2]) / (Cpt[2] **2)
        J[i][1] = -K[i][i] * (R[1][i] * Cpt[2] - Cpt[i] * R[1][2]) / (Cpt[2] **2)
        J[i][2] = -K[i][i] * (R[2][i] * Cpt[2] - Cpt[i] * R[2][2]) / (Cpt[2] **2)
        J[i][3] = K[i][i] * (np.dot(dev_r.T, Wpt_r.T)[i] * Cpt[2] - Cpt[i] * np.dot(dev_r.T, Wpt_r.T)[2]) / (Cpt[2] **2)
        J[i][4] = K[i][i] * (np.dot(dev_p.T, Wpt_r.T)[i] * Cpt[2] - Cpt[i] * np.dot(dev_p.T, Wpt_r.T)[2]) / (Cpt[2] **2)
        J[i][5] = K[i][i] * (np.dot(dev_y.T, Wpt_r.T)[i] * Cpt[2] - Cpt[i] * np.dot(dev_y.T, Wpt_r.T)[2]) / (Cpt[2] **2)

    #------------------

    return J


def rpy_from_dcm(R):

    rpy = np.zeros((3, 1))

    # Roll.
    rpy[0] = np.arctan2(R[2, 1], R[2, 2])

    # Pitch.
    sp = -R[2, 0]
    cp = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    if np.abs(cp) > 1e-15:
        rpy[1] = np.arctan2(sp, cp)
    else:
        # Gimbal lock...
        rpy[1] = np.pi / 2

        if sp < 0:
            rpy[1] = -rpy[1]

    # Yaw.
    rpy[2] = np.arctan2(R[1, 0], R[0, 0])

    return rpy

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
