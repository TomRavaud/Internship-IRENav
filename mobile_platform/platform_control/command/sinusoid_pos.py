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
    # pub_pitch = rospy.Publisher("/mobile_platform/pitch_joint_position_controller/command",
    #                       Float64, queue_size=1)

    # Wait for 2 seconds to make sure connexions between this node
    # and position controllers are done
    rospy.Rate(0.5).sleep()
    
    # Rate at which messages are published on the topic 
    publishing_rate = 50
    rate = rospy.Rate(publishing_rate)

    # Instantiate a Float64 object to store the position value that will be sent
    # to the platform
    position = Float64()
    # position2 = Float64()

    # Set the amplitude (radians) and the frequency (Hz) of the sinusoid
    A = 40*np.pi/180
    f = 0.05
    # f2 = 0.05
    
    # Get the current time to start the sinusoid at 0
    time = rospy.get_time()
    
    while not rospy.is_shutdown():
        # position.data = amplitude*np.cos(2*np.pi*frequency*(rospy.get_time()))
        # position.data = amplitude*np.sin(2*np.pi*frequency*(rospy.get_time()-time))
        position.data = sinusoid(rospy.get_time() - time, A, f)
        # position2.data = sinusoid(rospy.get_time() - time, A, f2)
        print(position.data)

        # Publish the position
        pub_roll.publish(position)
        # pub_pitch.publish(position2)

        rate.sleep()
