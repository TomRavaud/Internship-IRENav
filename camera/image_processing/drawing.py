import cv2
import numpy as np


def show_image(image):
    # Display the image in the window
    cv2.imshow("Preview", image)
    
    # Wait for 3 ms (for a key press) before automatically destroying
    # the current window
    cv2.waitKey(3)

def draw_points(image, points):
    # Draw a red circle on the image for each corner
    for point in points:
        cv2.circle(image, tuple(point), radius=3, 
                    color=(0, 0, 255), thickness=-1)
        
    return image

# TODO: Doc to write again
def draw_axes(image, rotation_vec, t, K, dist=None):
    """
    Draw a 6dof axis (XYZ -> RGB) in the given rotation and translation
    :param img - rgb numpy array
    :rotation_vec - euler rotations, numpy array of length 3,
                    use cv2.Rodrigues(R)[0] to convert from rotation matrix
    :t - 3d translation vector, in meters (dtype must be float)
    :K - intrinsic calibration matrix , 3x3
    :dist - optional distortion coefficients, numpy array of length 4. If None distortion is ignored.
    """
    # image = image.astype(np.float32)
    dist = np.zeros(4, dtype=float) if dist is None else dist
    
    # in meters
    points = np.float32([[0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.3], [0, 0, 0]]).reshape(-1, 3)
    
    axis_points, _ = cv2.projectPoints(points, rotation_vec, t, K, dist)
    
    image = cv2.line(image, tuple(axis_points[3].ravel()), tuple(axis_points[0].ravel()), (255, 0, 0), 3)
    image = cv2.line(image, tuple(axis_points[3].ravel()), tuple(axis_points[1].ravel()), (0, 255, 0), 3)
    image = cv2.line(image, tuple(axis_points[3].ravel()), tuple(axis_points[2].ravel()), (0, 0, 255), 3)
    
    return image
