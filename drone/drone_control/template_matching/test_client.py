#!/usr/bin/env python3

import rospy

import actionlib

from drone_control.msg import PoseEstimationAction, PoseEstimationGoal, PoseEstimationResult, PoseEstimationFeedback

# Initialize the node
rospy.init_node("template_tracking_action_client")

# Initialize the template tracking action client
template_tracking_client = actionlib.SimpleActionClient(
    "drone/TemplateTracking", PoseEstimationAction)

# Block until the action server is ready
template_tracking_client.wait_for_server()

# Set the action goal
template_tracking_goal = PoseEstimationGoal()
template_tracking_goal.estimating_pose = True

# Send the goal to the server
template_tracking_client.send_goal(template_tracking_goal)

