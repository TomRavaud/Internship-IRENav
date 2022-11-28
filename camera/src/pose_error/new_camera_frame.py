#!/usr/bin/env python3

import rospy
import tf2_msgs.msg
import geometry_msgs.msg
import tf_conversions
import numpy as np


if __name__ == '__main__':
    # Initialize the node
    rospy.init_node('camera_frame_broadcaster')
    
    # Initialize a publisher to the tf topic
    pub_camera_tf = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage,
                                    queue_size=1)
    
    # Rate at which messages are published on the topic 
    publishing_rate = 50.
    rate = rospy.Rate(publishing_rate)
    
    # Instantiate a ROS transform object
    camera_transform = geometry_msgs.msg.TransformStamped()
    
    # Indicate parent and new child's frame id
    camera_transform.header.frame_id = "camera1/camera_link"
    camera_transform.child_frame_id = "camera1/camera_frame_oriented"
    
    # Set the translation
    camera_transform.transform.translation.x = 0.0
    camera_transform.transform.translation.y = 0.0
    camera_transform.transform.translation.z = 0.0

    # Set the rotation
    quat = tf_conversions.transformations.quaternion_from_euler(-np.pi/2,
                                                                0.,
                                                                -np.pi/2)
    camera_transform.transform.rotation.x = quat[0]
    camera_transform.transform.rotation.y = quat[1]
    camera_transform.transform.rotation.z = quat[2]
    camera_transform.transform.rotation.w = quat[3]
    
    while not rospy.is_shutdown():
        # Set the stamp to the current time
        camera_transform.header.stamp = rospy.Time.now()
        
        # Convert the ROS transform message to a tf format message
        camera_transform_message = tf2_msgs.msg.TFMessage([camera_transform])
        
        # Publish the message
        pub_camera_tf.publish(camera_transform_message)
       
        rate.sleep()
