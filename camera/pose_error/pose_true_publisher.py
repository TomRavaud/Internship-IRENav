#!/usr/bin/env python3

# ROS - Python library
import rospy

import tf2_ros

# Import useful ROS types
from geometry_msgs.msg import TransformStamped

if __name__ == "__main__":
    # Initialize the node
    rospy.init_node("pose_listener_publisher")
    
    # Initialize a publisher to the pose_true topic
    # The topic is marked as latched, meaning that every nodes which will
    # subscribe to it will receive the last message sent on it (and there will
    # not be an error because the next message is not arrived yet when we need
    # it)
    pub_pose = rospy.Publisher("pose_true", TransformStamped, queue_size=1,
                               latch=True)
    
    # Instantiate a buffer and a tf listener
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    
    # Rate at which messages are published on the topic 
    publishing_rate = 50
    rate = rospy.Rate(publishing_rate) 
    
    while not rospy.is_shutdown():
        # Use a try/except to get rid of errors occurring at the beginning
        try:
            # Get the transform between the camera frame and the board frame
            transform = tfBuffer.lookup_transform(
                "camera1/camera_frame_oriented",
                "mobile_platform/board_upper_side", rospy.Time())
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rate.sleep()
            continue
        
        pub_pose.publish(transform)

        rate.sleep()
 