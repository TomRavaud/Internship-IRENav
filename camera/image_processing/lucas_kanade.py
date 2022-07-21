import numpy as np
import cv2


def points_next_position_lk(image, old_image, old_points):
    # Parameters for Lucas-Kanade optical flow computation
    #
    # maxLevel : number of levels of the pyramid
    # if 0, pyramids are not used (single level),
    # if 1, 2 different levels,
    # etc..
    lk_params = dict(winSize=(15, 15), maxLevel=0,
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

def sparse_optical_flow(old_points, points, time_difference):
    # Compute the optical flow, ie the velocity of image points
    optical_flow = (points - old_points)/time_difference
    return optical_flow
