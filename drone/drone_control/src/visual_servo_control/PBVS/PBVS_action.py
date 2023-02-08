#!/usr/bin/env python3

# ROS - Python librairies
import rospy
import actionlib

# Import useful ROS types
from std_msgs.msg import Float32
from sensor_msgs.msg import Image, CameraInfo, JointState
from geometry_msgs.msg import TransformStamped, TwistStamped

# Custom messages
from drone_control.msg import PBVSAction, PBVSGoal, PBVSResult, PBVSFeedback,\
    DecisionAction, DecisionGoal, DecisionResult, DecisionFeedback

# Python librairies
import numpy as np
import cv2

# My modules
from image_utils import optical_flow as of, drawing as dw, conversions, SIFT_detection_matching as sift
import PBVS_velocity_command as vc


class DroneControl:
    def __init__(self):
        """Constructor of the class
        """
        # Initialize the subscriber to the estimated velocity (ArduPilot EKF)
        # topic
        self.sub_velocity = rospy.Subscriber(
            "mavros/local_position/velocity_body", TwistStamped,
            self.callback_velocity, queue_size=1)
        
        # Initialize the subscriber to the platform joint states topic
        self.sub_platform_joints = rospy.Subscriber(
            "mobile_platform/joint_states", JointState,
            self.callback_platform_joints, queue_size=1)
        
        # Initialize the subscriber to the pose_estimate topic
        self.sub_pose = rospy.Subscriber("pose_estimate", TransformStamped,
                                         self.callback_pose, queue_size=1)

        # Initialize a publisher to the drone velocity topic
        self.pub_velocity = rospy.Publisher(
            "mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=1)
        
        # Initialize the PBVS server
        self.PBVS_server = actionlib.SimpleActionServer("drone/PBVS",
                                                        PBVSAction,
                                                        self.do_PBVS,
                                                        False)
        # Start the server
        self.PBVS_server.start()
        

    def do_PBVS(self, goal):
        
        # Compute the (constant) interaction matrix
        L = vc.compute_interaction_matrix()
        
        while not self.PBVS_server.is_preempt_requested():
            
            # Set a 30Hz frequency
            rospy.Rate(30).sleep()
        
        PBVS_result = PBVSResult()
        PBVS_result.is_drone_stabilized = True
        self.PBVS_server.set_preempted(PBVS_result)
    
    
    # def callback_velocity(self, msg):
    #     self.measured_velocity[0] = msg.twist.linear.x
    #     self.measured_velocity[1] = -msg.twist.linear.y
    #     self.measured_velocity[2] = -msg.twist.linear.z
    #     self.measured_velocity[3] = -msg.twist.angular.z
        
    # def callback_platform_joints(self, msg):
    #     self.platform_velocity[0] = msg.velocity[2]
    #     self.platform_velocity[1] = msg.velocity[3]
    #     self.platform_velocity[2] = msg.velocity[4]
    #     self.platform_velocity[3] = msg.velocity[5]
        
    def callback_pose(self, msg):
        camera_platform_transform = conversions.tf_to_transform_matrix(msg)
        theta = cv2.Rodrigues(camera_platform_transform[:3, :3])[0]


# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    # Declare the node
    rospy.init_node("PBVS")

    # Instantiate an object
    drone_control = DroneControl()

    # Run the node until Ctrl + C is pressed
    rospy.spin()
 