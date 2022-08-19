import cv2
import numpy as np


def show_image(image):
    """Display a given image

    Args:
        image (cv::Mat): a basic OpenCV image
    """
    # Display the image in a window
    cv2.imshow("Preview", image)
    
    # Wait for 3 ms (for a key press) before automatically destroying
    # the current window
    cv2.waitKey(3)

def draw_points(image, points):
    """Draw some points on an image

    Args:
        image (cv::Mat): an OpenCV image
        points (ndarray (N, 2)): a set of points

    Returns:
        cv::Mat: the modified image
    """
    # Draw a red circle on the image for point
    for point in points:
        cv2.circle(image, tuple(point), radius=3,
                   color=(0, 0, 255), thickness=-1)
        
    return image

def draw_axes(image, axes_points_image):
    """Draw a coordinate frame from 4 points

    Args:
        image (cv::Mat): an OpenCV image
        axes_points (ndarray (4, 2)): the 4 points' coordinates in the image
        coordinate system

    Returns:
        cv::Mat: the modified image
    """
    # Convert points coordinates to int to correspond to pixel values
    axes_points_image = axes_points_image.astype(int)
    
    # Link the points with 3 lines to represent the axes
    image = cv2.line(image, tuple(axes_points_image[3]),
                     tuple(axes_points_image[0]), (255, 0, 0), 3)
    image = cv2.line(image, tuple(axes_points_image[3]),
                     tuple(axes_points_image[1]), (0, 255, 0), 3)
    image = cv2.line(image, tuple(axes_points_image[3]),
                     tuple(axes_points_image[2]), (0, 0, 255), 3)
    
    return image
