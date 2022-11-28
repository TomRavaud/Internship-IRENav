#! /usr/bin/env python3

import rospy
from mavros_msgs.srv import SetMode, SetModeRequest, CommandBool, CommandBoolRequest, CommandTOL, CommandTOLRequest


if __name__ == "__main__":
    # Initialize the node
    rospy.init_node("takeoff")

    rospy.wait_for_service("mavros/set_mode")
    rospy.wait_for_service("mavros/cmd/arming")
    rospy.wait_for_service("mavros/cmd/takeoff")
    
    set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)
    arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
    takeoff_client = rospy.ServiceProxy("mavros/cmd/takeoff", CommandTOL)
    
    # Set the flight mode to guided
    mode = SetModeRequest()
    mode.custom_mode = "GUIDED"
    set_mode_client.call(mode)
    
    # Arm the throttles
    arming = CommandBoolRequest()
    arming.value = True
    arming_client.call(arming)
    
    # rospy.Rate(0.2).sleep()
    
    # Take-off
    takeoff = CommandTOLRequest()
    takeoff.altitude = 4
    takeoff_client.call(takeoff)
