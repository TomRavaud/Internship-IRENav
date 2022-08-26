#!/usr/bin/env python3

import rospy
import tf2_msgs.msg
import geometry_msgs.msg


# class FixedTFBroadcaster:

#     def __init__(self):
#         self.pub_tf = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=1)

#         while not rospy.is_shutdown():
#             # Run this loop at about 10Hz
#             rospy.sleep(0.1)

#             t = geometry_msgs.msg.TransformStamped()
#             t.header.frame_id = "mobile_platform/board_link"
#             t.header.stamp = rospy.Time.now()
#             t.child_frame_id = "mobile_platform/board_upper_side"
#             t.transform.translation.x = 0.0
#             t.transform.translation.y = 0.0
#             t.transform.translation.z = 0.025

#             t.transform.rotation.x = 0.0
#             t.transform.rotation.y = 0.0
#             t.transform.rotation.z = 0.0
#             t.transform.rotation.w = 1.0

#             tfm = tf2_msgs.msg.TFMessage([t])
#             self.pub_tf.publish(tfm)

# if __name__ == '__main__':
#     rospy.init_node('board_frame_broadcaster')
#     tfb = FixedTFBroadcaster()

#     rospy.spin()

if __name__ == '__main__':
    # Initialize the node
    rospy.init_node('board_frame_broadcaster')
    
    # Initialize a publisher to the tf topic
    pub_board_tf = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage,
                                    queue_size=1)
    
    # Rate at which messages are published on the topic 
    publishing_rate = 50. # Same that robot_state_publisher's rate
    rate = rospy.Rate(publishing_rate)
    
    # Instantiate a ROS transform object
    board_transform = geometry_msgs.msg.TransformStamped()
    
    # Indicate parent and new child's frame id
    board_transform.header.frame_id = "mobile_platform/board_link"
    board_transform.child_frame_id = "mobile_platform/board_upper_side"
    
    # Set the translation
    board_transform.transform.translation.x = 0.0
    board_transform.transform.translation.y = 0.0
    board_transform.transform.translation.z = 0.025

    # Set the rotation
    board_transform.transform.rotation.x = 0.
    board_transform.transform.rotation.y = 0.
    board_transform.transform.rotation.z = 0.
    board_transform.transform.rotation.w = 1.
    
    while not rospy.is_shutdown():
        # Set the stamp to the current time
        board_transform.header.stamp = rospy.Time.now()
        
        # Convert the ROS transform message to a tf format message
        board_transform_message = tf2_msgs.msg.TFMessage([board_transform])
        
        # Publish the message
        pub_board_tf.publish(board_transform_message)
       
        rate.sleep()
