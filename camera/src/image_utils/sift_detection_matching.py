import cv2
import numpy as np
from packaging import version


def detect_and_match(image1, image2, nb_points=10):
    # OpenCV version must be higher than 4.4.0
    # If it is not : pip install opencv-python
    assert version.parse(cv2.__version__) > version.parse("4.4.0"),\
        "OpenCV version must be higher than 4.4.0.\
        Try 'pip install opencv-python'."
    
    # Convert to grayscale
    # image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    # image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find the keypoints and their descriptor within the two images
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
    
    keypoints1 = cv2.KeyPoint.convert(keypoints1)
    keypoints2 = cv2.KeyPoint.convert(keypoints2)

    # Brute-force matching with L2 norm and
    # cross check to ensure better results
    brute_force_matcher = cv2.BFMatcher.create(normType=cv2.NORM_L2, crossCheck=True)
    matches = brute_force_matcher.match(descriptors1, descriptors2)

    # Sort matches
    matches = sorted(matches, key = lambda x:x.distance)
    
    nb_matches = len(matches)
    
    if nb_points > nb_matches:
        nb_points = nb_matches
        
    matched_keypoints1 = np.zeros((nb_points, 2))
    matched_keypoints2 = np.zeros((nb_points, 2))
    
    for i in range(nb_points):
        matched_keypoints1[i] = keypoints1[matches[i].queryIdx]
        matched_keypoints2[i] = keypoints2[matches[i].trainIdx]
    
    # return keypoints1, keypoints2
    return matched_keypoints1, matched_keypoints2
    