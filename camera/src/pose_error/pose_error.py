#!/usr/bin/env python3

# ROS - Python library
import rospy

import tf2_ros
import numpy as np
import cv2

# Import useful ROS types
from geometry_msgs.msg import TransformStamped

# My modules
import error_function as ef

class PoseError:
    def __init__(self):
        self.T = np.zeros((1000, 5))
        self.counter = 0
        
        # Define a variable which will store the true transform between
        # the camera frame and the board frame
        self.transform_true = None
        
        # Initialize the subscriber to the pose_true topic
        self.sub_pose_true = rospy.Subscriber("pose_true", TransformStamped,
                                              self.callback_pose_true)
        
        # Initialize the subscriber to the pose_estimate topic
        self.sub_pose_estimate = rospy.Subscriber("pose_estimate", TransformStamped,
                                         self.callback_pose_estimate)
    
    def callback_pose_true(self, msg):
        """Function called each time a new ROS TransformStamped message is
        received in the pose_true topic

        Args:
            msg (ROS TransformStamped): the true transform computed from
            the tf topic
        """
        self.transform_true = msg
        
    def callback_pose_estimate(self, msg):
        """Function called each time a new ROS TransformStamped message is
        received on the pose_estimate topic

        Args:
            msg (ROS TransformStamped): a computed transform between the
            camera and the board frames
        """
        if self.transform_true is not None:
            # Convert the true transform to a numpy array
            HTM1 = ef.tf_to_transform_matrix(self.transform_true.transform)
        
            # Convert the transform estimate to a numpy array
            HTM2 = ef.tf_to_transform_matrix(msg.transform)
        
            # Compute errors on translation and rotation between the true value
            # and the estimated one
            error_translation, error_rotation = ef.error_function(HTM1, HTM2)

            print("The error on the translation is "
                  f"{round(error_translation, 4)} m\n"
                  "The error on the rotation is "
                  f"{round(error_rotation, 4)} rad"
                  "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
            
            if self.counter < 1000:
                # Get simulation time
                self.T[self.counter, 0] = rospy.get_time()
                
                # Get errors on translation and rotation
                self.T[self.counter, 1] = round(error_translation, 4)
                self.T[self.counter, 2] = round(error_rotation, 4)
                
                # Get true translation and rotation
                self.T[self.counter, 3] = np.linalg.norm(HTM1[:3, -1])
                self.T[self.counter, 4] = np.linalg.norm(cv2.Rodrigues(HTM1[:3, :3])[0])
                
                
                
            if self.counter == 1000:
                np.save("pose_error.npy", self.T)
                
            self.counter += 1
                

 
if __name__ == "__main__":
    rospy.init_node("pose_error")
    
    # Instantiate an object
    pose_error = PoseError() 

    rospy.spin() 
 