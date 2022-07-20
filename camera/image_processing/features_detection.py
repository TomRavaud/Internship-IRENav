# TODO: Change to OOP

import numpy as np
import cv2


def compute_harris_score(image):
    # Convert the image to grayscale 
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
    
    # Intensities should be float32 type 
    gray = np.float32(gray)
    
    # Compute the Harris score of each pixel 
    # cornerHarris(img, blockSize, ksize, k)
    harris_score = cv2.cornerHarris(gray, 2, 3, 0.04)
    
    return harris_score

def corners_detection(harris_score, threshold):
    # Identify corners in the image given a threshold
    corners = np.flip(np.column_stack(
        np.where(harris_score > threshold * harris_score.max())))
    
    return corners 

# TODO: Maybe to modify, differentiate displaying the images and the corners
def show_points(image, points): 
    # Draw a red circle on the image for each corner
    for point in points:
        cv2.circle(image, tuple(point), radius=3, 
                    color=(0, 0, 255), thickness=-1)
        
    # Display the image in the window
    cv2.imshow("Preview", image)
    
    # Wait for 3 ms (for a key press) before automatically destroying
    # the current window
    cv2.waitKey(3)
