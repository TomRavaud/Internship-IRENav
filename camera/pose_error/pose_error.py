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
        self.sub_pose = rospy.Subscriber("pose_estimate", TransformStamped, self.callback_pose)
        
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
 
    def callback_pose(self, msg):
        self.trans = self.tfBuffer.lookup_transform("camera1/camera_frame_oriented", "mobile_platform/board_upper_side", rospy.Time())
        HTM1 = ef.tf_to_transform_matrix(self.trans.transform)
        
        HTM2 = ef.tf_to_transform_matrix(msg.transform)
        
        error_translation, error_rotation = ef.error_function(HTM1, HTM2)

        print(f"The error on the translation is {round(error_translation, 4)} m\nThe error on the rotation is {round(error_rotation, 4)} rad\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

 
if __name__ == '__main__':
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
 