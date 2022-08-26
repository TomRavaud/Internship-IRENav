import numpy as np
import cv2


def points_next_position_lk(image, old_image, old_points):
    """Find points coordinates in the current image from their coordinates
    in the previous one

    Args:
        image (cv::Mat): the current image
        old_image (cv::Mat): the previous image
        old_points (ndarray (N, 2)): points coordinates in the previous image

    Returns:
        ndarrays (N', 2): points coordinates that have been found in the
        current image and their homologues in the previous one
    """
    # Parameters for Lucas-Kanade optical flow computation
    #
    # maxLevel : number of levels of the pyramid
    # if 0, pyramids are not used (single level),
    # if 1, 2 different levels,
    # etc..
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                     10, 0.03))

    # Convert the current and the last frames to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    old_gray = cv2.cvtColor(old_image, cv2.COLOR_BGR2GRAY)

    # Reshape the key-points' array
    # (should be of dimension (nb of points, 1, 2))
    # and convert its type to float32
    # to work with OpenCV calcOpticalFlowPyrLK function
    old_points = old_points.reshape(-1, 1, 2)
    old_points = np.float32(old_points)

    # Compute optical flow using Lucas-Kanade method
    points, status, err = cv2.calcOpticalFlowPyrLK(old_gray,
                                                   gray,
                                                   old_points, None,
                                                   **lk_params)

    # Keep only good points
    # (status = 1 if the flow for the corresponding feature has been found)
    # Note : it reshapes the arrays to the dimension (nb of points, 2)
    good_points = points[status == 1]
    good_old_points = old_points[status == 1]

    return good_points, good_old_points

def sparse_displacement(old_points, points):
    """Compute the displacement of some points

    Args:
        old_points (ndarray (N, 2)): the previous coordinates of the points
        points (ndarray (N, 2)): the current coordinates of the points

    Returns:
        ndarray (N, 2): the array storing the displacement of each point
    """
    return points - old_points

def sparse_optical_flow(old_points, points, time_difference):
    """Compute the optical flow at given image points

    Args:
        old_points (ndarray (N, 2)): the previous points' coordinates
        points (ndarray (N, 2)): the current points' coordinates
        time_difference (float): the time difference between
        the two sets of points

    Returns:
        ndarray (N, 2): the optical flow values at those points
    """
    # Compute the optical flow, ie the velocity of image points
    optical_flow = sparse_displacement(old_points, points)/time_difference
    
    return optical_flow
