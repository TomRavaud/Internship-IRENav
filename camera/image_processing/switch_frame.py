import numpy as np


def apply_rigid_motion(points, R, T):
    """Give points' coordinates in a new frame obtained after rotating (R)
    and translating (T) the current one

    Args:
        points (ndarray (N, 2)): a set of points
        R (ndarray (3, 3)): a rotation matrix
        T (ndarray (3,)): a translation vector

    Returns:
        ndarray (N, 2): points new coordinates
    """
    # Number of points we want to move
    nb_points = np.shape(points)[0]
    
    # Use homogenous coordinates
    homogeneous_points = np.ones((nb_points, 4))
    homogeneous_points[:, :-1] = points
    
    # Concatenate the rotation matrix and the translation vector
    homogeneous_matrix = np.zeros((3, 4))
    homogeneous_matrix[:, :-1], homogeneous_matrix[:, -1] = R, T
    
    # Compute points coordinates after the rigid motion
    points_new = np.dot(homogeneous_points, np.transpose(homogeneous_matrix))
    
    return points_new

def camera_frame_to_image(points, K):
    """Compute points coordinates in the image frame from their coordinates in
    the camera frame

    Args:
        points (ndarray (N, 2)): a set of points
        K (ndarray (3, 3)): the internal calibration matrix

    Returns:
        ndarray (N, 2): points image coordinates
    """
    # Project the points onto the image plan, the obtained coordinates are
    # defined up to a scaling factor
    points_projection = np.dot(points, np.transpose(K))
    
    # Get the points' coordinates in the image frame dividing by the third
    # coordinate
    points_image = points_projection[:, :2]/points_projection[:, 2][:, np.newaxis]
 
    return points_image
 