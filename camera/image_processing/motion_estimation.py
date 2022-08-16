import numpy as np


def velocity_estimation(optical_flow, old_points, f, nu0, mu0):
    
    # Number of points for which the optical flow has been successfully
    # computed
    nb_points = np.shape(old_points)[0]
    
    # Check we have enough points (3) to estimate the 6 velocities
    assert nb_points >= 3, "Need to have at least 3 detected points"
    
    # Change the image's origin from the top left corner to the center
    points = np.copy(old_points) - np.array([mu0, nu0])

    # Get the key-points' coordinates
    mu, nu = points[:, 0], points[:, 1]

    # Compute the lines of the matrix M obtained by derivation of the ideal optical flow
    M0 = np.stack((-mu*nu/f, mu**2/f + f, -nu), axis=1)  # Even lines
    M1 = np.stack((nu**2/f + f, mu*nu/f, mu), axis=1)  # Odd lines

    # Arrange these lines to form the matrix M
    M = np.empty((2*nb_points, 3))
    M[0::2, :] = M0
    M[1::2, :] = M1
    
    # Reshape the optical flow vector
    # print(np.concatenate((old_points, optical_flow), axis=1))
    print(f)
    optical_flow = optical_flow.reshape(2*nb_points,)
    
    # Estimate the velocities by solving the linear least squares problem
    # (it computes the Moore Penrose pseudo inverse of M using its SVD)
    rot_velocities = np.linalg.lstsq(M, optical_flow, rcond=None)[0]
    # print(rot_velocities)

    return rot_velocities
