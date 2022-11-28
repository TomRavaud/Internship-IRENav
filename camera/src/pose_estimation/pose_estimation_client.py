#!/usr/bin/env python3

import rospy
import cv2

import actionlib
import cv_bridge

from camera.msg import PoseEstimationAction, PoseEstimationGoal


# Initialize the node
rospy.init_node("pose_estimation_action_client")

# Initialize the template tracking action client
pose_estimation_client = actionlib.SimpleActionClient(
    "PoseEstimation", PoseEstimationAction)

# Block until the action server is ready
pose_estimation_client.wait_for_server()

# Set the action goal
pose_estimation_goal = PoseEstimationGoal()


pose_estimation_goal.estimating_pose = True

# Send the goal to the server
pose_estimation_client.send_goal(pose_estimation_goal)
