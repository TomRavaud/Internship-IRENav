import numpy as np

def compute_infinitesimal_rigid_motion(points_displacement, old_points, K, depth_image):
    """Compute the infinitesimal rigid motion of an object in the camera
    coordinate system from the displacement on the images of some of its
    points (at least 3) measured between 2 successive time points

    Args:
        points_displacement (ndarray (N, 2)): the displacement of the points
        in pixels
        old_points (ndarray (N, 2)): the position of the points in the
        previous image
        K (ndarray (3, 3)): the internal calibration matrix of the camera
        depth_image (cv::Mat): depth in meters in the camera frame of points
        corresponding to each image pixel

    Returns:
        ndarray (6,): the estimation of infinitesimal translation and rotation
    """
    # Get the coordinates of the image's origin
    mu0, nu0 = K[0, 2], K[1, 2]

    # Get the key-points' coordinates
    points = np.copy(old_points)
    mu, nu = points[:, 0], points[:, 1]
    
    #FIXME: inversion of mu and nu in the depth image ???
    # Associate those points with their depth in the camera frame
    zc = depth_image[tuple(nu.astype(int)), tuple(mu.astype(int))]
    # print(depth_image[int(nu[1]), int(mu[1])])
    
    #TODO: Keep non zeros ??
    # Detect NaN values in zc and remove the corresponding points
    zc[zc == 0] = np.nan
    non_nan = ~np.isnan(zc)
    zc, mu, nu, points_displacement = zc[non_nan], mu[non_nan], nu[non_nan], points_displacement[non_nan]
    
    # Number of points for which the optical flow has been successfully
    # computed and the depth has been obtained
    nb_points = np.count_nonzero(non_nan)
    # print("Nombre points : ", nb_points)
    # print(zc)
    
    # points = points[non_nan]
    # for point in points:
    #     cv2.circle(depth_image, tuple(point), radius=3,
    #                color=(0, 255, 0), thickness=-1)
        
    # cv2.imshow("Test", depth_image)
    
    # # Wait for 3 ms (for a key press) before automatically destroying
    # # the current window
    # cv2.waitKey(3)
    
    # Check we have enough points (3) to estimate the 6 motion parameters
    assert nb_points >= 3, "Need to have at least 3 detected points"
    
    # Change the image's origin from the top left corner to the center
    mu -= mu0
    nu -= nu0

    # Compute the lines of the matrix M obtained by derivation of the ideal
    # optical flow
    f = K[0, 0] # Get the focal length of the camera
    M0 = np.stack((f/zc, np.zeros_like(mu), -mu/zc,
                   -mu*nu/f, mu**2/f + f, -nu), axis=1)  # Even lines
    M1 = np.stack((np.zeros_like(mu), f/zc, -nu/zc,
                   -nu**2/f - f, mu*nu/f, mu), axis=1)  # Odd lines

    # Arrange these lines to form the matrix M
    M = np.empty((2*nb_points, 6))
    M[0::2, :] = M0
    M[1::2, :] = M1
    
    # Reshape the displacement vector to the shape needed to apply the lstsq
    # function
    points_displacement = points_displacement.reshape(2*nb_points,)
    
    # Estimate the motion by solving the linear least squares problem
    # (it computes the Moore Penrose pseudo inverse of M using its SVD)
    dmotion = np.linalg.lstsq(M, points_displacement, rcond=None)[0]
    
    return dmotion

def infinitesimal_rotation_matrix(dtheta):
    """Compute the infinitesimal rotation matrix from the rotation angles

    Args:
        dtheta (ndarray (3,)): infinitesimal rotation angles

    Returns:
        ndarray (3, 3): infinitesimal rotation matrix
    """
    dthetax, dthetay, dthetaz = dtheta
    
    # Compute the infinitesimal rotation matrix from the angles
    dR = np.array([[1, -dthetaz, dthetay],
                   [dthetaz, 1, -dthetax],
                   [-dthetay, dthetax, 1]])
    
    return dR
