#!/usr/bin/env python3

import rospy
import actionlib

# Import useful ROS types
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import TransformStamped

# Import custom messages
from drone_control.msg import DecisionAction, DecisionGoal, DecisionResult, DecisionFeedback

# Python libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

# Import my modules
from image_processing import conversions

class LandingDecision:
    def __init__(self):
        # Declare the variable which will store the transform between
        # the camera and the platform frames
        self.camera_platform_transform = np.zeros((3, 3))
        
        # Initialize the IBVS_error variable
        self.IBVS_error = 100
        
        # Initialize the subscriber to the pose_estimate topic
        self.sub_pose = rospy.Subscriber("pose_estimate", TransformStamped,
                                         self.callback_pose, queue_size=1)
        
        # Initialize the subscriber to the IBVS_error topic
        self.sub_IBVS_error = rospy.Subscriber("IBVS_error", Float32,
                                               self.callback_IBVS,
                                               queue_size=1)
        
        # Initialize the publisher to the landing decision topic        
        # self.pub_decision = rospy.Subscriber("is_ready_to_land", Bool,
        #                                      queue_size=1)
        
        # Initialize the decision action server
        self.decision_server = actionlib.SimpleActionServer("drone/Decision",
                                                             DecisionAction,
                                                             self.do_decide,
                                                             False)
        # Start the server
        self.decision_server.start()
        
    def do_decide(self, goal):
        
        # "Reactive" mode gives the order to land once a condition is reached
        if goal.mode == "reactive":
            # Set the rotation we have when platform and drone axes
            # are aligned
            R_target = np.array([[0, -1, 0],
                                 [-1, 0, 0],
                                 [0, 0, -1]])
            
            is_ready_to_land = False
            
            # file = open("record_drone_platform_angle.csv", "w")
            # writer = csv.writer(file)
            # header = ["timestamp", "angle"]
            
            # writer.writerow(header)
            
            L_roll = []
            
            # while not is_ready_to_land:
            for i in range(1000):
                # print(f"IBVS error : {self.IBVS_error}")
                is_stabilized = self.IBVS_error <= 30.
                # L_error.append(self.IBVS_error)
                
                # print(f"Transform matrix : {self.camera_platform_transform}")
                # theta = cv2.Rodrigues(self.camera_platform_transform[:3, :3])[0]
                
                # Extract the rotation matrix from the transform matrix
                R = self.camera_platform_transform[:3, :3]
                
                # Rotation between the target rotation and the estimated one
                R_error = np.dot(np.transpose(R), R_target)
                
                # Get Euler angles from the rotation matrix
                theta_error = cv2.Rodrigues(R_error)[0]
                
                # Pitch and roll angles as a single one
                drone_platform_angle = np.linalg.norm(theta_error[:2])
                
                # writer.writerow([i, drone_platform_angle])
                L_roll.append(theta_error[0])
                
                # print(self.camera_platform_transform[:3, :3])
                # print(drone_platform_angle)
                is_platform_horizontal = drone_platform_angle <= 0.05
                
                
                print(f"IBVS : {is_stabilized}")
                print(f"Horizontal : {is_platform_horizontal}")
                is_ready_to_land = is_stabilized and is_platform_horizontal
                
                # Send action feedback
                decision_feedback = DecisionFeedback()
                decision_feedback.IBVS_error = self.IBVS_error
                decision_feedback.drone_platform_angle = drone_platform_angle
                self.decision_server.publish_feedback(decision_feedback)
                
                rospy.Rate(10).sleep()
                
            # file.close()
            columns = ["Angle"]
            dataframe = pd.DataFrame(L_roll, columns=columns)
            dataframe.to_csv("record_drone_platform_angle.csv")
            
            # plt.figure()
            # plt.plot(range(200), L_error)
            # plt.show()
                
            decision_result = DecisionResult()
            decision_result.is_ready_to_land = True
            self.decision_server.set_succeeded(decision_result)
        
    def callback_pose(self, msg):
        self.camera_platform_transform = conversions.tf_to_transform_matrix(msg.transform)        
        
    def callback_IBVS(self, msg):
        self.IBVS_error = msg.data


if __name__ == "__main__":
    rospy.init_node("landing_decision")
    
    landing_decision = LandingDecision()
    
    rospy.spin()
