#!/usr/bin/env python3

# ROS - Python librairies
import rospy

# cv_bridge is used to convert ROS Image message type into OpenCV images
import cv_bridge

from sensor_msgs.msg import Image, CameraInfo 


# Python librairies
import cv2
import numpy as np


# My modules
import features_detection as fd
import lucas_kanade as lk
import motion_estimation as me
import pose_estimation as pe
import drawing as dw


class OpticalFlow:
    def __init__(self):
        
        # Set a boolean attribute to identify the first frame
        self.is_first_image = True
        
        # Declare some attributes which will be used to compute optical flow
        self.old_image = None
        self.old_points = None
        self.old_time = None
        
        # Initialize the set of 4 points which will be used to draw the axes
        self.axes_points = np.array([[0.3, 0, 0], [0, 0.3, 0],
                                     [0, 0, 0.3], [0, 0, 0]], dtype="float32")
        
        
        # Initialize the rotation and the translation
        self.theta = np.array([0., np.pi, 0.])
        self.T = np.array([0., 0., 1.475])
        self.R = np.array([[0, 1, 0],
                          [1, 0, 0],
                          [0, 0, -1]])
        
        # Declare a depth image attribute
        self.depth_image = None
        
        # Initialize the bridge between ROS and OpenCV images
        self.bridge = cv_bridge.CvBridge()
        
        # FIXME: Do I have to open a window here ?
        # Open a window in which camera images will be displayed
        # cv2.namedWindow("Preview", 1)
        
        # Extract only one message from the camera_info topic as the camera parameters
        # do not change
        camera_info = rospy.wait_for_message("camera1/camera_info", CameraInfo)
        
        # Get the internal calibration matrix of the camera
        self.K = camera_info.K
        self.K = np.reshape(np.array(self.K), (3, 3))
        
        # Initialize the subscriber to the camera images topic
        self.sub_image = rospy.Subscriber("camera1/image_raw", Image, self.callback_image)
        
        # Initialize the subscriber to the depth image topic
        self.sub_depth = rospy.Subscriber("camera1/image_raw_depth", Image, self.callback_depth)

    # TODO: Callback function, docstring 
    def callback_image(self, msg):
        
        #FIXME: Keep the time or not ?
        # Get the time the message was published
        time = msg.header.stamp
        
        # Convert the ROS Image into the OpenCV format
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Find the Harris' corners on the first frame
        if self.is_first_image:
            # Compute the Harris' score map
            harris_score = fd.compute_harris_score(image)
            
            # Set a threshold to avoid having too many corners,
            # its value depends on the image
            threshold = 0.7
            
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
            
            # Estimate the rotational relative velocities
            # rot_velocities = me.velocity_estimation(optical_flow, self.old_points, self.f, self.mu0, self.nu0)
            
            # Estimate the infinitesimal motion
            dmotion = pe.compute_infinitesimal_rigid_motion(optical_flow, self.old_points, self.K, self.depth_image)
            
            # Extract infinitesimal rotation and translation
            self.T = dmotion[:3]
            self.theta = dmotion[-3:]

            # Compute the infinitesimal rotation matrix from the angles
            self.R = pe.infinitesimal_rotation_matrix(self.theta)
            
        # Update old image and points
        self.old_image = image
        self.old_points = points
        
        # Update the time the last message was published
        self.old_time = time

        # Draw the 3D coordinates axis of the platform
        # image = dw.draw_axes(image, self.theta, self.T, self.K, self.axes_points)
        
        # Update axes points coordinates
        self.axes_points = pe.apply_rigid_motion(self.axes_points, self.R, self.T)
        
        axes_points_image = pe.camera_frame_to_image(self.axes_points, self.K)
        
        image = dw.draw_axes2(image, axes_points_image)
        
        # Draw points on the image
        image = dw.draw_points(image, points)
        
        # Display the image
        dw.show_image(image)
        
    def callback_depth(self, msg):
        # print(2)
        
        #TODO: To be cleaned
        # for p in pc2.read_points(msg, field_names = ("x", "y", "z"), skip_nans=True):
        #     print(" x : %f  y: %f  z: %f" %(p[0],p[1],p[2]))
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        # self.depth_image = image
        
        # points = np.array([[400, 400],
        #                    [550, 550]])
        # print(self.depth_image[(400, 0, 550, 0), (400, 0, 550, 0)])
        # image = dw.draw_points(image, points)
        # dw.show_image(self.depth_image)


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
