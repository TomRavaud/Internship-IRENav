#!/usr/bin/env python3

import rospy

# cv_bridge is used to convert ROS Image message type into OpenCV images
import cv_bridge

import cv2
import numpy as np

from sensor_msgs.msg import Image


class OpticalFlow:
    def __init__(self):
        # Initialize the bridge between ROS and OpenCV images
        self.bridge = cv_bridge.CvBridge()
        
        # Open a window in which camera images will be displayed
        cv2.namedWindow("Preview", 1)
        
        # This node subscribes to the camera images topic
        self.sub = rospy.Subscriber("camera1/image_raw", Image, self.callback)
    
    def compute_harris_score(self, image):
        # Convert the image to grayscale 
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
        
        # Intensities should be float32 type 
        gray = np.float32(gray)
        
        # Compute the Harris score of each pixel 
        # cornerHarris(img, blockSize, ksize, k)
        harris_score = cv2.cornerHarris(gray, 2, 3, 0.04)
        
        return harris_score
    
    def corners_detection(self, harris_score, threshold):
        # Identify corners in the image given a threshold
        corners = np.flip(np.column_stack(np.where(harris_score > threshold * harris_score.max())))
        
        return corners 
    
    def show_corners(self, image, corners): 
        # Draw a red circle on the image for each corner
        for corner in corners:
            cv2.circle(image, tuple(corner), radius=3, 
                       color=(0, 0, 255), thickness=-1)
            
        # Display the image in the window
        cv2.imshow("Preview", image)
        
        # Wait for 3 ms (for a key press) before automatically destroying
        # the current window
        cv2.waitKey(3)
        
    # TODO: Callback function, docstring 
    def callback(self, msg):
        # Convert the ROS Image into the OpenCV format
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        
        # Compute the harris score map
        harris_score = self.compute_harris_score(image)
        
        # Set a threshold to avoid having too many corners,
        # its value depends on the image
        threshold = 0.01
        
        # Extract corners coordinates
        corners = self.corners_detection(harris_score, threshold)
        
        # Display the image and the detected corners
        self.show_corners(image, corners)



# Declare the node
rospy.init_node("optical_flow")

# Instantiate an object
optical_flow = OpticalFlow()

# Run the node until Ctrl + C is pressed
rospy.spin()
