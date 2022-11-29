#!/usr/bin/env python3

import rospy

import actionlib

from camera.msg import StateEstimationAction, StateEstimationGoal


# Initialize the node
rospy.init_node("state_estimation_action_client")

# Initialize the state estimation action client
state_estimation_client = actionlib.SimpleActionClient(
    "StateEstimation", StateEstimationAction)

# Block until the action server is ready
state_estimation_client.wait_for_server()

# Set the action goal
state_estimation_goal = StateEstimationGoal()


state_estimation_goal.estimating_state = True

# Send the goal to the server
state_estimation_client.send_goal(state_estimation_goal)
