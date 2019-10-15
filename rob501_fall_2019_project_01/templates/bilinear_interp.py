import numpy as np
from numpy.linalg import inv

def bilinear_interp(I, pt):
    """
    Performs bilinear interpolation for a given image point.

    Given the (x, y) location of a point in an input image, use the surrounding
    4 pixels to conmpute the bilinearly-interpolated output pixel intensity.

    Note that images are (usually) integer-valued functions (in 2D), therefore
    the intensity value you return must be an integer (use round()).

    This function is for a *single* image band only - for RGB images, you will
    need to call the function once for each colour channel.

    Parameters:
    -----------
    I   - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).
    pt  - 2x1 np.array of point in input image (x, y), with subpixel precision.

    Returns:
    --------
    b  - Interpolated brightness or intensity value (whole number >= 0).
    """
    #--- FILL ME IN ---

    if pt.shape != (2, 1):
        raise ValueError('Point size is incorrect.')
    else:
        x = pt[1]
        y = pt[0] # the point we want to estimate
        x1 = int(np.floor(pt[1]))
        x2 = int(np.ceil(pt[1]))
        y1 = int(np.floor(pt[0]))
        y2 = int(np.ceil(pt[0])) # coordinates of surrounding 4 pixels
        Q11 = I[x1][y1]
        Q21 = I[x2][y1]
        Q12 = I[x1][y2]
        Q22 = I[x2][y2] # the pixel values of surrounding 4 points
        R1 = ((x2 - x)/(x2 - x1))*Q11 + ((x - x1)/(x2 - x1))*Q21
        R2 = ((x2 - x)/(x2 - x1))*Q12 + ((x - x1)/(x2 - x1))*Q22
        b = int(np.round(((y2 - y)/(y2 - y1))*R1 + ((y - y1)/(y2 - y1))*R2)) # the pixel value of desired point

    #------------------

    return b
