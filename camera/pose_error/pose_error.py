#!/usr/bin/env python3

# ROS - Python library
import rospy

import tf2_ros

# Import useful ROS types
from geometry_msgs.msg import TransformStamped

# My modules
import error_function as ef

class PoseError:
    def __init__(self):
        # Initialize the subscriber to the pose_estimate topic
        self.sub_pose = rospy.Subscriber("pose_estimate", TransformStamped,
                                         self.callback_pose)
        
        # Instantiate a buffer and a tf listener
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
 
    def callback_pose(self, msg):
        """Function called each time a new ROS TransformStamped message is
        received on the pose_estimate topic

        Args:
            msg (ROS TransformStamped): a computed transform between the
            camera and the board frames
        """
        # Get the transform between the camera frame and the board frame
        self.trans = self.tfBuffer.lookup_transform(
            "camera1/camera_frame_oriented",
            "mobile_platform/board_upper_side", rospy.Time())
        
        # Convert the Transform type to a numpy array
        HTM1 = ef.tf_to_transform_matrix(self.trans.transform)
        
        # Convert the transform estimate to a numpy array
        HTM2 = ef.tf_to_transform_matrix(msg.transform)
        
        # Compute errors on translation and rotation between the true value
        # and the estimated one
        error_translation, error_rotation = ef.error_function(HTM1, HTM2)

        #TODO: continue the string on the next line
        print("The error on the translation is "
              f"{round(error_translation, 4)} m\n"
              "The error on the rotation is "
              f"{round(error_rotation, 4)} rad"
              "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

 
if __name__ == "__main__":
    rospy.init_node("pose_error")
    
    # Instantiate an object
    pose_error = PoseError()    

    # tfBuffer = tf2_ros.Buffer()
    # listener = tf2_ros.TransformListener(tfBuffer)

    # rate = rospy.Rate(10.0)
    # try:
    #     trans = tfBuffer.lookup_transform("camera1/camera_frame_oriented",
    #                                       "mobile_platform/board_upper_side", rospy.Time())
    # except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
    #     rate.sleep()

    rospy.spin() 
    # msg.transform
 