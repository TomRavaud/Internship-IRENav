import numpy as np

def compute_infinitesimal_rigid_motion(optical_flow, old_points, K, depth_image):
    
    # Number of points for which the optical flow has been successfully
    # computed
    nb_points = np.shape(old_points)[0]
    
    # Check we have enough points (3) to estimate the 6 velocities
    assert nb_points >= 3, "Need to have at least 3 detected points"
    
    # Change the image's origin from the top left corner to the center
    mu0, nu0 = K[0, 2], K[1, 2] # Get the coordinates of the image's origin
    # points = np.copy(old_points) - np.array([mu0, nu0])

    # Get the key-points' coordinates
    points = np.copy(old_points)
    mu, nu = points[:, 0], points[:, 1]
    
    zc = depth_image[tuple(mu.astype(int)), tuple(nu.astype(int))]
    # print(zc)
    # print(np.shape(mu), np.shape(nu), np.shape(1/zc))
    
    mu -= mu0
    nu -= nu0

    # dtheta = np.array([0., 0., 0.])
    
    #FIXME: Introduce zc !
    # Compute the lines of the matrix M obtained by derivation of the ideal optical flow
    f = K[0, 0] # Get the focal length of the camera
    M0 = np.stack((f/zc, np.zeros_like(mu), -mu/zc, -mu*nu/f, mu**2/f + f, -nu), axis=1)  # Even lines
    M1 = np.stack((np.zeros_like(mu), f/zc, -nu/zc, -nu**2/f - f, mu*nu/f, mu), axis=1)  # Odd lines

    # Arrange these lines to form the matrix M
    M = np.empty((2*nb_points, 6))
    M[0::2, :] = M0
    M[1::2, :] = M1
    
    # Reshape the optical flow vector
    # print(np.concatenate((old_points, optical_flow), axis=1))
    optical_flow = optical_flow.reshape(2*nb_points,)
    
    # Estimate the motion by solving the linear least squares problem
    # (it computes the Moore Penrose pseudo inverse of M using its SVD)
    dmotion = np.linalg.lstsq(M, optical_flow, rcond=None)[0]
    
    # print(dmotion)

    return dmotion

def apply_infinitesimal_rigid_motion(points, dtheta, dT):
    # Compute the infinitesimal rotation matrix from the angles
    dthetax, dthetay, dthetaz = dtheta
    
    dR = np.array([[1, -dthetaz, dthetay],
                   [dthetaz, 1, -dthetax],
                   [-dthetay, dthetax, 1]])
    
    # Number of points we want to move
    n = np.shape(points)[0]
    homogeneous_points = np.ones((n, 4))
    homogeneous_points[:, :-1] = points
    
    # Concatenate the rotation matrix and the translation vector
    homogeneous_matrix = np.zeros((3, 4))
    homogeneous_matrix[:, :-1], homogeneous_matrix[:, -1] = dR, dT
    
    # Compute points coordinates after the rigid motion
    new_points = np.dot(homogeneous_points, np.transpose(homogeneous_matrix))
    
    return new_points

def apply_rigid_motion(points, R, T):
    
    # Number of points we want to move
    nb_points = np.shape(points)[0]
    
    # Use homogenous coordinates
    homogeneous_points = np.ones((nb_points, 4))
    homogeneous_points[:, :-1] = points
    
    # Concatenate the rotation matrix and the translation vector
    homogeneous_matrix = np.zeros((3, 4))
    homogeneous_matrix[:, :-1], homogeneous_matrix[:, -1] = R, T
    
    # Compute points coordinates after the rigid motion
    new_points = np.dot(homogeneous_points, np.transpose(homogeneous_matrix))
    
    return new_points

def infinitesimal_rotation_matrix(dtheta):
     # Compute the infinitesimal rotation matrix from the angles
    dthetax, dthetay, dthetaz = dtheta
    
    dR = np.array([[1, -dthetaz, dthetay],
                   [dthetaz, 1, -dthetax],
                   [-dthetay, dthetax, 1]])
    
    return dR

def camera_frame_to_image(points, K):
    # Project the points onto the image plan, the obtained coordinates are
    # defined up to a scaling factor
    points_projection = np.dot(points, np.transpose(K))
    print(points_projection)
    
    # Get the points' coordinates in the image frame dividing by the third coordinate
    points_projection = points_projection[:, :2]/points_projection[:, 2][:, np.newaxis]
    
    return points_projection
 