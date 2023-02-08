#!/usr/bin/env python3

# ROS - Python librairies
import rospy

# A library to deal with actions
import actionlib

# Import useful ROS types
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped

from camera.msg import StateEstimationAction, StateEstimationResult,\
    StateEstimationFeedback

# Python librairies
import numpy as np
import cv2

# My modules
from image_utils import switch_frame as sf, drawing as dw, conversions, error_function as ef
from kalman import kalman_stationary


class StateEstimation:
    def __init__(self):
        """Constructor of the class
        """
        # Load the linear system matrices
        # (This script must be run from the top-level src directory
        # for the matrices to be found, else change their relative path)
        self.A = np.load("./camera/src/state_estimation/A.npy")
        self.C = np.load("./camera/src/state_estimation/C.npy")
        
        # Load the pre-computed gain matrix to use in the stationary Kalman filter
        self.K = np.load("./camera/src/state_estimation/K.npy")
        
        
        # Declare the rigid transform from the camera frame to the platform frame
        self.HTM_cp_estimated = None
        
        # Declare the rigid transform from the world frame to the platform frame
        self.HTM_wp_true = None
        
        
        # Get the transform between the world and the camera (the camera is supposed to be fixed)
        world_camera_transform = rospy.wait_for_message("world_camera_transform", TransformStamped)
        # Convert it to a homogeneous transform matrix
        self.HTM_wc_true = conversions.tf_to_transform_matrix(world_camera_transform.transform)
        

        # Initialize the subscriber to camera-platform pose topic
        self.sub_transform_cp = rospy.Subscriber("pose_estimate",
                                                 TransformStamped,
                                                 self.callback_pose_estimate,
                                                 queue_size=1)
        
        # Initialize the subscriber to world-platform pose topic
        self.sub_transform_wp = rospy.Subscriber("world_platform_transform",
                                                 TransformStamped,
                                                 self.callback_transform_wp,
                                                 queue_size=1)
        
        # Initialize the state estimation action server
        self.state_estimation_server = actionlib.SimpleActionServer("StateEstimation",
                                                                    StateEstimationAction,
                                                                    self.do_state_estimation,
                                                                    False)
        # Start the server
        self.state_estimation_server.start()
        

    def do_state_estimation(self, goal):
        
        # Initialisation (a priori knowledge on x)
        x_pred = np.zeros((36, 1))
        
        # FIXME: display covariance matrix on the x, y position ?
        
        dt = 0.1
        
        self.T = np.zeros((1000, 5))
        self.counter = 0
        
        
        while not self.state_estimation_server.is_preempt_requested():
            
            # Get the true transform between the world and the platform
            HTM_wp_true = self.HTM_wp_true
            
            # Compute an estimate of the transform between the world and the
            # platform
            HTM_wp_estimated = np.dot(self.HTM_wc_true, self.HTM_cp_estimated)
            T_wp_estimated = HTM_wp_estimated[:3, 3]
            R_wp_estimated = HTM_wp_estimated[:3, :3]
            theta = cv2.Rodrigues(R_wp_estimated)[0]
            
            # Fill the observation vector
            y = np.array([T_wp_estimated[0], T_wp_estimated[1], T_wp_estimated[2],
                          theta[0, 0], theta[1, 0], theta[2, 0], 0, 0, 0, 0, 0, 0])
            y = y[:, None]
            
            if self.counter < 500:
                # State estimation
                x_pred = kalman_stationary(x_pred, y, self.C, 0, self.A, self.K)
            else:
                x_pred = np.dot(self.A, x_pred)
            
            # Get the prediction of the transform between the world and the platform
            T_wp_pred = x_pred[:3, 0]
            angles = x_pred[3:6, 0]
            R_wp_pred = cv2.Rodrigues(angles)[0]
            HTM_wp_pred = sf.transform_matrix(R_wp_pred, T_wp_pred)
            
            # Compute errors on translation and rotation between the true value
            # and the estimated ones
            error_translation, error_rotation = ef.error_function(HTM_wp_true, HTM_wp_estimated)
            error_translation_kalman, error_rotation_kalman = ef.error_function(HTM_wp_true, HTM_wp_pred)
            
            print(f"Translation error : {error_translation}")
            print(f"Rotation error : {error_rotation}")
            print(f"Translation error bis : {error_translation_kalman}")
            print(f"Rotation error bis : {error_rotation_kalman}")
            print("")
            
            if self.counter < 1000:
                # Get simulation time
                self.T[self.counter, 0] = rospy.get_time()
                
                # Get errors on translation and rotation
                # self.T[self.counter, 1] = round(error_translation, 4)
                # self.T[self.counter, 2] = round(error_translation_kalman, 4)
                self.T[self.counter, 1] = np.linalg.norm(HTM_wp_true[:3, -1])
                self.T[self.counter, 2] = np.linalg.norm(HTM_wp_estimated[:3, -1])
                
                # Get true translation and rotation
                self.T[self.counter, 3] = np.linalg.norm(HTM_wp_pred[:3, -1])
                # self.T[self.counter, 4] = np.linalg.norm(cv2.Rodrigues(HTM1[:3, :3])[0])
            
            if self.counter == 1000:
                np.save("pose_error.npy", self.T)
            
            self.counter += 1
            print(self.counter)
            
            
            # Set the (time-dependent) evolution matrix
            rospy.Rate(10).sleep()


    def callback_pose_estimate(self, msg):
        self.HTM_cp_estimated = conversions.tf_to_transform_matrix(msg.transform)

    def callback_transform_wp(self, msg):
        self.HTM_wp_true = conversions.tf_to_transform_matrix(msg.transform)


# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    # Declare the node
    rospy.init_node("state_estimation")

    # Instantiate an object
    state_estimation = StateEstimation()

    # Run the node until Ctrl + C is pressed
    rospy.spin()
 