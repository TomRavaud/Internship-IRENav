import numpy as np


def compute_interaction_matrix(old_points, K):
    # TODO:  doc
    
    # Get the coordinates of the image's origin
    mu0, nu0 = K[0, 2], K[1, 2]

    # Get the key-points' coordinates
    points = np.copy(old_points)
    mu, nu = points[:, 0], points[:, 1]
    
    # Associate those points with their depth in the camera frame
    # zc = depth_image[tuple(nu.astype(int)), tuple(mu.astype(int))]
    # print(zc)
    
    nb_points = np.shape(old_points)[0]
    zc = np.ones(nb_points)*1.475
    
    # Detect NaN values in zc and remove the corresponding points
    # zc[zc == 0] = np.nan
    # non_nan = ~np.isnan(zc)
    # zc, mu, nu = zc[non_nan], mu[non_nan], nu[non_nan]
    
    # Number of points for which the optical flow has been successfully
    # computed and the depth has been obtained
    # nb_points = np.count_nonzero(non_nan)
    
    # Check we have enough points (3) to estimate the 6 motion parameters
    assert nb_points >= 4, "Need to have at least 3 detected points to avoid\
    singularities"
    
    # Change the image's origin from the top left corner to the center
    mu -= mu0
    nu -= nu0

    # Compute the lines of the interaction matrix L obtained by derivation of the ideal
    # optical flow
    f = K[0, 0] # Get the focal length of the camera
    L0 = -np.stack((f/zc, np.zeros_like(mu), -mu/zc, -nu), axis=1)  # Even lines
    L1 = -np.stack((np.zeros_like(mu), f/zc, -nu/zc, mu), axis=1)  # Odd lines

    # Arrange these lines to form the matrix L
    L = np.empty((2*nb_points, 4))
    L[0::2, :] = L0
    L[1::2, :] = L1
    
    return L

def velocity_command(L, points, old_points, target_points,
                     platform_vel, lamb=0.5):
    
    # print(dt)
    
    # Compute the error between current and target points positions
    error = points - target_points
    
    # Compute the error at the previous time step
    # old_error = old_points - target_points
    
    nb_points = np.shape(points)[0]
    
    # Reshape error vectors to the shape needed to apply the lstsq
    # function
    error = error.reshape(2*nb_points,)
    # old_error = old_error.reshape(2*nb_points,)
    
    # print((error - old_error)/dt)
    
    # Compute the partial derivative of the error with respect to the time to
    # take the motion of the platform into account
    # error_partial_derivative = 0
    # error_partial_derivative = (error - old_error)/dt - np.dot(L, measured_velocity)
    # print("//////////////////////////\n")
    # print((error - old_error)[:10]/dt)
    # print(np.dot(L, measured_velocity)[:10])
    # print(measured_velocity)
    # print("//////////////////////////\n")
    
    # Compute the drone velocity command by solving the linear least squares
    # problem Ax = b where x is the velocity
    # (it computes the Moore Penrose pseudo inverse of L using its SVD)
    velocity_cmd = np.linalg.lstsq(L, -lamb*error, rcond=None)[0]
    velocity_cmd[0] -= platform_vel[1]
    velocity_cmd[1] -= platform_vel[0]
    velocity_cmd[2] -= platform_vel[2]
    velocity_cmd[3] -= platform_vel[3]
    
    # velocity_cmd = np.linalg.lstsq(L, -lamb*error - error_partial_derivative, rcond=None)[0]
    
    # print(error_partial_derivative)
    
    return velocity_cmd
