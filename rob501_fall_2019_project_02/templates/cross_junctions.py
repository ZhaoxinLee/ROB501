import numpy as np
from scipy.ndimage.filters import *
# import matplotlib.pyplot as plt

def cross_junctions(I, bounds, Wpts):
    """
    Find cross-junctions in image with subpixel accuracy.

    The function locates a series of cross-junction points on a planar
    calibration target, where the target is bounded in the image by the
    specified quadrilateral. The number of cross-junctions identified
    should be equal to the number of world points.

    Note also that the world and image points must be in *correspondence*,
    that is, the first world point should map to the first image point, etc.

    Parameters:
    -----------
    I       - Single-band (greyscale) image as np.array (e.g., uint8, float).
    bounds  - 2x4 np.array, bounding polygon (clockwise from upper left).
    Wpts    - 3xn np.array of world points (in 3D, on calibration target).

    Returns:
    --------
    Ipts  - 2xn np.array of cross-junctions (x, y), relative to the upper
            left corner of I. These should be floating-point values.
    """
    #--- FILL ME IN ---

    Ipts = np.zeros((2, 48))

    # use DLT homography to rotate the checker board
    Rctpts = np.array([[0,np.size(I,1),np.size(I,1),0],[0,0,np.size(I,0),np.size(I,0)]])
    H,A = dlt_homography(Rctpts, bounds)
    J = I.copy()

    # perform bilinear interpolation
    for i in range(0,np.size(I,0)):
        for j in range(0,np.size(I,1)):
            x = np.dot(H,(np.array([j,i,1])))[0]
            y = np.dot(H,(np.array([j,i,1])))[1]
            w = np.dot(H,(np.array([j,i,1])))[2]
            pt = np.array([[x/w,y/w]]).T
            b = bilinear_interp(I,pt)
            J[i][j] = b

    J = gaussian_filter(J,4)

    # use Harris corner detecting method to detect corners
    J, corners = harris_corner(J)

    # find the centroid of each cluster to determine the cross junctions
    Xjunc = find_centroids(corners)

    # convert the coordinates of cross junctions back to initial image using homography
    for i in range(Xjunc.shape[0]):
        x = np.dot(H, (np.array([Xjunc[i][1], Xjunc[i][0], 1])))[0]
        y = np.dot(H, (np.array([Xjunc[i][1], Xjunc[i][0], 1])))[1]
        w = np.dot(H, (np.array([Xjunc[i][1], Xjunc[i][0], 1])))[2]
        # print(x,y,w)
        Ipts[0][i] = x/w
        Ipts[1][i] = y/w
    #     I[int(Ipts[1][i])][int(Ipts[0][i])]=255
    #
    # plt.imshow(J, cmap = None, vmin = 0, vmax = 255)
    # plt.show()
    # plt.imshow(I, cmap = None, vmin = 0, vmax = 255)
    # plt.show()

    #------------------

    return Ipts

def dlt_homography(I1pts, I2pts):

    x = I1pts[0]
    y = I1pts[1]
    u = I2pts[0]
    v = I2pts[1]

    # concatenate the matrix A using 4 points correspondences
    A = np.empty([0,9], int)
    for i in range(4):
        A = np.append(A,[[-x[i],-y[i],-1,0,0,0,u[i]*x[i],u[i]*y[i],u[i]],[0,0,0,-x[i],-y[i],-1,v[i]*x[i],v[i]*y[i],v[i]]],axis=0)

    # calculate H using null space
    B = A[:,0:8]
    C = -A[:,8]
    H = np.dot(np.linalg.inv(B),C)
    H = np.append(H,1)
    H = np.reshape(H,(3,3))

    #------------------

    return H, A

def bilinear_interp(I, pt):

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
        if y2 - y1 != 0 or x2-x1 != 0:
            R1 = ((x2 - x)/(x2 - x1))*Q11 + ((x - x1)/(x2 - x1))*Q21
            R2 = ((x2 - x)/(x2 - x1))*Q12 + ((x - x1)/(x2 - x1))*Q22
            b = int(np.round(((y2 - y)/(y2 - y1))*R1 + ((y - y1)/(y2 - y1))*R2)) # the pixel value of desired point
        else:
            b = I[int(x)][int(y)]

    #------------------

    return b

def harris_corner(I):
    Ix = np.zeros((np.size(I,0), np.size(I,1)))
    Iy = np.zeros((np.size(I,0), np.size(I,1)))

    # compute gradient for all the pixels (ignore the border)
    for i in range(1,np.size(I,0)-1):
        for j in range(1,np.size(I,1)-1):
            Ix[i][j] = (int(I[i][j+1])-int(I[i][j-1]))/2
            Iy[i][j] = (int(I[i+1][j])-int(I[i-1][j]))/2

    # define the harris corner detector
    Wsize = 2 # window size
    IxIy_sum = 0
    corners = []
    for i in range(1,(np.size(I,0)-Wsize+1)):
        for j in range(1,(np.size(I,1)-Wsize+1)):
            W_Ix = Ix[i:(i+Wsize),j:(j+Wsize)]
            W_Iy = Iy[i:(i+Wsize),j:(j+Wsize)]
            Ix_sqr_sum = np.square(W_Ix).sum()
            Iy_sqr_sum = np.square(W_Iy).sum()
            for m in range(Wsize):
                for n in range(Wsize):
                    IxIy_sum = IxIy_sum + W_Ix[m][n]*W_Iy[m][n]
            M = np.array([[Ix_sqr_sum,IxIy_sum],[IxIy_sum,Iy_sqr_sum]])
            eigval,eigvec = np.linalg.eig(M)
            lambda1 = eigval[0]
            lambda2 = eigval[1]
            R = lambda1*lambda2 -0.04*(lambda1-lambda2)*(lambda1-lambda2)
            IxIy_sum = 0

            if R>30:
                # filter the points near the image border
                if i - 0 < 50 or np.size(I,0) - i < 50:
                    pass
                elif j - 0 < 50 or np.size(I,1) - j < 50:
                    pass
                else:
                    I[i][j] = 255
                    corners.append([i,j])
    corners = np.array(corners) # here we get all the corner points distributed as clusters near cross junctions
    return I, corners

def find_centroids(corners):
    sum_x = np.zeros((1,corners.shape[0]))
    sum_y = np.zeros((1,corners.shape[0]))
    centroid_x = np.zeros((1,corners.shape[0]))
    centroid_y = np.zeros((1,corners.shape[0]))

    # for each corner point, we try to find all the neighbor points within the distance of 15 pixels
    # and then compute the average of sum(x) and sum(y), which is the coordinate of centroid point
    counter = 0
    for i in range(corners.shape[0]):
        for j in range(corners.shape[0]):
            if (corners[i][0] - corners[j][0])**2 + (corners[i][1] - corners[j][1])**2 > 225:
                pass
            else:
                sum_x[0][i] += corners[j][0]
                sum_y[0][i] += corners[j][1]
                counter += 1
        centroid_x[0][i] = sum_x[0][i]/counter
        centroid_y[0][i] = sum_y[0][i]/counter
        counter = 0
    centroid = np.vstack((centroid_x,centroid_y)).T # the centroid point corresponding to each corner point

    # count the occurance of each centroid, the centroids most commonly occurred tend to be the centroids of clusters
    unique, counts = np.unique(centroid, return_counts=True, axis=0)

    # select the top 48 most frequent centroids
    centroids = []
    for i in range(counts.size):
        # here we observe the results and choose 6 as our best threshold which can filter all the noise
        if counts[i] > 6:
            centroids.append(unique[i])
    centroids = np.array(centroids)
    centroids_ordered = centroids.copy()

    # order the centroids from top left
    for i in range(6):
        y = centroids[8*i:8*i+8,[1]].T
        index = np.argsort(y)
        centroids_ordered[8*i:8*i+8] = centroids[index[0]+8*i]

    return centroids_ordered
