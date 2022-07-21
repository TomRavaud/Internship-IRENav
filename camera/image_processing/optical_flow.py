#!/usr/bin/env python3

# ROS - Python librairies
import rospy

# cv_bridge is used to convert ROS Image message type into OpenCV images
import cv_bridge

from sensor_msgs.msg import Image


# Python librairies
import cv2
import numpy as np


# My modules
import features_detection as fd
import lucas_kanade as lk


class OpticalFlow:
    def __init__(self):
        # Set a boolean attribute to identify the first frame
        self.is_first_image = True
        
        # Declare some attributes which will be used to compute optical flow
        self.old_image = None
        self.old_points = None
        self.old_time = None
        
        # Initialize the bridge between ROS and OpenCV images
        self.bridge = cv_bridge.CvBridge()
        
        # FIXME: Do I have to open a window here ?
        # Open a window in which camera images will be displayed
        # cv2.namedWindow("Preview", 1)
        
        # Initialize the subscriber to the camera images topic
        self.sub = rospy.Subscriber("camera1/image_raw", Image, self.callback)
        
    
    # TODO: Callback function, docstring 
    def callback(self, msg):
        
        # Convert the ROS Image into the OpenCV format
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        
        # Get the time the message was published
        time = msg.header.stamp

        # Find the Harris' corners on the first frame
        if self.is_first_image:
            # Compute the Harris' score map
            harris_score = fd.compute_harris_score(image)
            
            # Set a threshold to avoid having too many corners,
            # its value depends on the image
            threshold = 0.1
            
            # Extract corners coordinates
            points = fd.corners_detection(harris_score, threshold)
            
            self.is_first_image = False

        # Compute the optical flow from the second frame
        else:
            # The optical flow might have not been found for some points
            # we need to update old_points to compute the difference between
            # the current points and these
            points, self.old_points = lk.points_next_position_lk(image,
                                                              self.old_image,
                                                              self.old_points)

            # Get the duration between two published messages
            dt = (time - self.old_time).to_sec()
            
            # Compute optical flow between the current points and old ones
            optical_flow = lk.sparse_optical_flow(self.old_points, points, dt)

        # Display the image and key-points
        fd.show_points(image, points)

        # Update old image and points
        self.old_image = image
        self.old_points = points
        
        # Update the time the last message was published
        self.old_time = time


# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    # Declare the node
    rospy.init_node("optical_flow")

    # Instantiate an object
    optical_flow = OpticalFlow()

    # Run the node until Ctrl + C is pressed
    rospy.spin()
