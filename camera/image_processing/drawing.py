import cv2
import numpy as np


def show_image(image, window_name="Preview"):
    """Display a given image

    Args:
        image (cv::Mat): a basic OpenCV image
    """
    # Display the image in a window
    cv2.imshow(window_name, image)
    
    # Wait for 3 ms (for a key press) before automatically destroying
    # the current window
    cv2.waitKey(3)

def draw_points(image, points, color=(0, 0, 255)):
    """Draw some points on an image

    Args:
        image (cv::Mat): an OpenCV image
        points (ndarray (N, 2)): a set of points

    Returns:
        cv::Mat: the modified image
    """
    # Convert points coordinates to int to correspond to pixel values
    points = points.astype(int)
    
    # Draw a red circle on the image for point
    for point in points:
        cv2.circle(image, tuple(point), radius=3,
                   color=color, thickness=-1)
        
    return image

def draw_axes(image, axes_points_image, colors=[(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
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
                     tuple(axes_points_image[0]), colors[0], 3) # X axis (red)
    image = cv2.line(image, tuple(axes_points_image[3]),
                     tuple(axes_points_image[1]), colors[1], 3) # Y axis (green)
    image = cv2.line(image, tuple(axes_points_image[3]),
                     tuple(axes_points_image[2]), colors[2], 3) # Z axis (blue)
    
    return image
