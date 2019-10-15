import numpy as np

def histogram_eq(I):
    """
    Histogram equalization for greyscale image.

    Perform histogram equalization on the 8-bit greyscale intensity image I
    to produce a contrast-enhanced image J. Full details of the algorithm are
    provided in the Szeliski text.

    Parameters:
    -----------
    I  - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).

    Returns:
    --------
    J  - Contrast-enhanced greyscale intensity image, 8-bit np.array (i.e., uint8).
    """
    #--- FILL ME IN ---

    # Verify I is grayscale.
    if I.dtype != np.uint8:
        raise ValueError('Incorrect image format!')
    else:
        hist,bins = np.histogram(I,256,[0,256]) # there are 256 bins from 0 to 255
        cdf = hist.cumsum() # calculate cumulative distribution functions (cdf) for all pixel intensities (v)
        h = []
        for i in range(256):
            h.append(int(round((cdf[i] - cdf.min())*255/(I.size-cdf.min()),2))) # calculte the equalized pixel intensities using the general histogram equalization formula
        h = np.array(h) # convert from list to array
        J = h[I] # equalize the image
    #------------------

    return J
