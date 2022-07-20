#! /usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Float64

rospy.init_node("sinusoid_pos")

# The queue_size argument allows to synchronise publisher and subscriber
# frequencies
pub_roll = rospy.Publisher("/mobile_platform/roll_joint_position_controller/command",
                      Float64, queue_size=1)
# pub_pitch = rospy.Publisher("/mobile_platform/pitch_joint_position_controller/command",
#                       Float64, queue_size=1)

# Rate at which messages are published on the topic 
publishing_rate = 100
rate = rospy.Rate(publishing_rate)

# TODO: Define a function to compute the command
# Instantiate a Float64 object to store the position value that will be sent
# to the platform
position = Float64()

# Set the amplitude (radians) and the frequency (Hz) of the sinusoid
amplitude = 20*np.pi/180
frequency = 0.5

# time = rospy.get_time()

while not rospy.is_shutdown():
    position.data = amplitude*np.cos(2*np.pi*frequency*(rospy.get_time()))
    # position.data = amplitude*np.cos(2*np.pi*frequency*(rospy.get_time()-time))
    
    # Publish the position
    pub_roll.publish(position)
    # pub_pitch.publish(position)
     
    rate.sleep()
