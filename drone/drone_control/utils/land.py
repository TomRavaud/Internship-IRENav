#! /usr/bin/env python3

import rospy
from mavros_msgs.srv import SetMode, SetModeRequest


if __name__ == "__main__":
    # Initialize the node
    rospy.init_node("takeoff")

    rospy.wait_for_service("mavros/set_mode")
    
    set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)
    
    # Set the flight mode to land
    mode = SetModeRequest()
    mode.custom_mode = "LAND"
    set_mode_client.call(mode)
    