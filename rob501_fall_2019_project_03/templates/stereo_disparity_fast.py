import numpy as np
from numpy.linalg import inv
from scipy.ndimage.filters import *

def stereo_disparity_fast(Il, Ir, bbox, maxd):
    """
    Fast stereo correspondence algorithm.

    This function computes a stereo disparity image from left stereo
    image Il and right stereo image Ir. Only disparity values within
    the bounding box region are evaluated.

    Parameters:
    -----------
    Il    - Left stereo image, m x n pixel np.array, greyscale.
    Ir    - Right stereo image, m x n pixel np.array, greyscale.
    bbox  - 2x2 np.array, bounding box, relative to left image, from top left
            corner to bottom right corner (inclusive).
    maxd  - Integer, maximum disparity value; disparities must be within zero
            to maxd inclusive (i.e., don't search beyond rng)

    Returns:
    --------
    Id  - Disparity image (map) as np.array, same size as Il, greyscale.
    """
    # Hints:
    #
    #  - Loop over each image row, computing the local similarity measure, then
    #    aggregate. At the border, you may replicate edge pixels, or just avoid
    #    using values outside of the image.
    #
    #  - You may hard-code any parameters you require in this function.
    #
    #  - Use whatever window size you think might be suitable.
    #
    #  - Don't optimize for runtime (too much), optimize for clarity.

    #--- FILL ME IN ---

    # Your code goes here.

    maxd = 63

    # define the kernal which sums all the pixels in the window
    k = np.ones((9,9))

    # shift the right image one column per time and compute the SAD
    for d in range(maxd):
        Ir = np.hstack((np.zeros((Ir.shape[0],1)),Ir[:,:(Ir.shape[1]-1)]))
        diff = np.abs(Il-Ir)
        SAD = convolve(diff,k, mode='constant', cval=0.0)

        # find the minimum SAD for each point and substitute the corresponding disparity
        if d == 0:
            minSAD = SAD
            Id = np.zeros((Ir.shape[0],Ir.shape[1]))
        else:
            for i in range(Ir.shape[0]):
                for j in range(Ir.shape[1]):
                    if SAD[i][j] < minSAD[i][j]:
                        minSAD[i][j] = SAD[i][j]
                        Id[i][j] = d

    #------------------

    return Id
