import numpy as np
import cv2


def compute_harris_score(image):
    """Compute the Harris score map from an image

    Args:
        image (cv::Mat): an OpenCV image

    Returns:
        ndarray (image size): the score map of the image
    """
    # Convert the image to grayscale 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    
    # Intensities should be float32 type 
    gray = np.float32(gray)
    
    # Compute the Harris score of each pixel 
    # cornerHarris(img, blockSize, ksize, k)
    harris_score = cv2.cornerHarris(gray, 2, 3, 0.04)
    
    return harris_score

def corners_detection(harris_score, threshold):
    """Detect corners in a score map given a threshold

    Args:
        harris_score (ndarray): an image score map
        threshold (float): a threshold used to identify corners
        (the larger it is, the fewer points there will be)

    Returns:
        ndarray (N, 2): the array of corners' coordinates
    """
    # Identify corners in the image given a threshold
    corners = np.flip(np.column_stack(
        np.where(harris_score > threshold * harris_score.max())))
    
    return corners 
