# Billboard hack script file.
import numpy as np
from matplotlib.path import Path
from imageio import imread, imwrite
import matplotlib.pyplot as plt

from dlt_homography import dlt_homography
from bilinear_interp import bilinear_interp
from histogram_eq import histogram_eq

def billboard_hack():
    """
    Hack and replace the billboard!

    Parameters:
    -----------

    Returns:
    --------
    Ihack  - Hacked RGB intensity image, 8-bit np.array (i.e., uint8).
    """
    # Bounding box in Y & D Square image.
    bbox = np.array([[404, 490, 404, 490], [38,  38, 354, 354]])

    # Point correspondences.
    Iyd_pts = np.array([[416, 485, 488, 410], [40,  61, 353, 349]])
    Ist_pts = np.array([[2, 218, 218, 2], [2, 2, 409, 409]])

    Iyd = imread('../billboard/yonge_dundas_square.jpg')
    Ist = imread('../billboard/uoft_soldiers_tower_dark.png')

    Ihack = np.asarray(Iyd)
    Ist = np.asarray(Ist)

    #--- FILL ME IN ---

    # Let's do the histogram equalization first.
    Jst = histogram_eq(Ist)

    # Compute the perspective homography we need...
    H,A = dlt_homography(Ist_pts,Iyd_pts)

    # Main 'for' loop to do the warp and insertion -
    # this could be vectorized to be faster if needed!
    polygon = []
    for i in range(4):
        polygon.append((Iyd_pts[0][i],Iyd_pts[1][i]))
    path = Path(polygon) # compute the path of polygon

    # do inverse transform for the warped image
    for i in range(404,490):
        for j in range(38,354):
            if path.contains_points([[i,j]]):
                x = np.dot(np.linalg.inv(H),(np.array([i,j,1])))[0]
                y = np.dot(np.linalg.inv(H),(np.array([i,j,1])))[1]
                w = np.dot(np.linalg.inv(H),(np.array([i,j,1])))[2]

                # perform bilinear interpolation and insert the warped image
                pt = np.array([[x/w,y/w]]).T
                b = bilinear_interp(Jst,pt)
                Ihack[j][i] = b

    # # You may wish to make use of the contains_points() method
    # available in the matplotlib.path.Path class!

    #------------------

    # plt.imshow(Ihack)
    # plt.show()
    # imwrite('billboard_hacked.png', Ihack);

    return Ihack
