#!/usr/bin/env python3

import rospy

from std_msgs.msg import Float64
# Import our custom service
from platform_control.srv import JointsReset


def callback_reset(request):
    """Fonction run each time the service is called

    Args:
        request (std_msgs/bool): a boolean which indicates whether joints
        values need to be reset

    Returns:
        std_msgs/string: an information message
    """
    if request.reset:
        # Set all the joints values to 0
        sub_tx.publish(0)
        sub_ty.publish(0)
        sub_tz.publish(0)
        sub_yaw.publish(0)
        sub_pitch.publish(0)
        sub_roll.publish(0)
        
        return "Joints values have been reset"
    
    return "Joints values have not been reset"
    

# Init the node
rospy.init_node("joints_reset")

# Initialize publishers to joints controllers topics
sub_tx = rospy.Publisher(
    "/mobile_platform/tx_joint_position_controller/command",
    Float64, queue_size=1)
sub_ty = rospy.Publisher(
    "/mobile_platform/ty_joint_position_controller/command",
    Float64, queue_size=1)
sub_tz = rospy.Publisher(
    "/mobile_platform/tz_joint_position_controller/command",
    Float64, queue_size=1)
sub_roll = rospy.Publisher(
    "/mobile_platform/roll_joint_position_controller/command",
    Float64, queue_size=1)
sub_pitch = rospy.Publisher(
    "/mobile_platform/pitch_joint_position_controller/command",
    Float64, queue_size=1)
sub_yaw = rospy.Publisher(
    "/mobile_platform/yaw_joint_position_controller/command",
    Float64, queue_size=1)

# Initialize a service to allow setting simultaneously all joints values to 0
service = rospy.Service("reset_joints", JointsReset, callback_reset)

rospy.spin()
