import numpy as np
from numpy.linalg import inv, lstsq

def saddle_point(I):
    """
    Locate saddle point in an image patch.

    The function identifies the subpixel centre of a cross-junction in the
    image patch I, by fitting a hyperbolic paraboloid to the patch, and then
    finding the critical point of that paraboloid.

    Note that the location of 'p' is relative to (-0.5, -0.5) at the upper
    left corner of the patch, i.e., the pixels are treated as covering an
    area of one unit square.

    Parameters:
    -----------
    I  - Single-band (greyscale) image patch as np.array (e.g., uint8, float).

    Returns:
    --------
    pt  - 2x1 np.array, subpixel location of saddle point in I (x, y coords).
    """
    #--- FILL ME IN ---

    m, n = I.shape
    Z = [] # Z is a column matrix of "dependent variables"
    A = [] # A is the "independent variables" matrix
    for i in range(m):
        for j in range(n):
            row = [j*j, i*j, i*i, j, i, 1]
            A.append(row)
            Z.append(I[i][j])
    Z = np.array(Z) # with a shape of 400*1
    A = np.array(A) # with a shape of 400*6
    alpha,beta,gamma,delta,epsilon,zeta = lstsq(A, Z, rcond=None)[0]

    # calculate the subpixel location using the equation in the paper
    B = np.array([[2*alpha,beta],[beta,2*gamma]])
    C = np.array([[delta],[epsilon]])
    pt = -np.dot(inv(B),C)

    #------------------

    return pt
