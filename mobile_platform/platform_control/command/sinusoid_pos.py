#! /usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Float64

def sinusoid(t, amplitude, frequency):
    """Compute the value at time t of a sinusoidal signal

    Args:
        t (float): time point
        amplitude (float): amplitude of the signal
        frequency (float): frequency of the signal

    Returns:
        float: value at time t of the sinusoidal signal
    """
    return amplitude*np.sin(2*np.pi*frequency*t)


if __name__ == "__main__":
    # Initialize the node
    rospy.init_node("sinusoid_pos")

    # Initialize a publisher to the roll joint position command topic
    # The queue_size argument allows to synchronise publisher and subscriber
    # frequencies
    pub_roll = rospy.Publisher("/mobile_platform/roll_joint_position_controller/command", 
                               Float64, queue_size=1)
    pub_pitch = rospy.Publisher("/mobile_platform/pitch_joint_position_controller/command",
                          Float64, queue_size=1)
    pub_yaw = rospy.Publisher("/mobile_platform/yaw_joint_position_controller/command",
                              Float64, queue_size=1)
    pub_tx = rospy.Publisher("/mobile_platform/tx_joint_position_controller/command",
                              Float64, queue_size=1)
    pub_ty = rospy.Publisher("/mobile_platform/ty_joint_position_controller/command",
                              Float64, queue_size=1)
    pub_tz = rospy.Publisher("/mobile_platform/tz_joint_position_controller/command",
                              Float64, queue_size=1)

    # Wait for 2 seconds to make sure connexions between this node
    # and position controllers are done
    rospy.Rate(0.5).sleep()
    
    # Rate at which messages are published on the topic 
    publishing_rate = 50
    rate = rospy.Rate(publishing_rate)

    # Instantiate a Float64 object to store the position value that will be sent
    # to the platform
    position_roll = Float64()
    position_pitch = Float64()
    position_yaw = Float64()
    position_tx = Float64()
    position_ty = Float64()
    position_tz = Float64()

    # Set the amplitude (radians) and the frequency (Hz) of the sinusoid
    A = 10*np.pi/180
    f_roll = 0.05
    f_pitch = 0.025
    f_yaw = 0.05
    f_tx = 0.025
    f_ty = 0.025
    f_tz = 0.05
    
    # Get the current time to start the sinusoid at 0
    time = rospy.get_time()
    
    while not rospy.is_shutdown():
        position_roll.data = sinusoid(rospy.get_time() - time, A, f_roll)
        position_pitch.data = sinusoid(rospy.get_time() - time, A, f_pitch)
        position_yaw.data = sinusoid(rospy.get_time() - time, A, f_yaw)
        position_tx.data = sinusoid(rospy.get_time() - time, A, f_tx)
        position_ty.data = sinusoid(rospy.get_time() - time, A, f_ty)
        position_tz.data = sinusoid(rospy.get_time() - time, A, f_tz)

        # Publish the position
        pub_roll.publish(position_roll)
        # pub_pitch.publish(position_pitch)
        # pub_yaw.publish(position_yaw)
        # pub_tx.publish(position_tx)
        # pub_ty.publish(position_ty)
        # pub_tz.publish(position_tz)

        rate.sleep()
