#!/usr/bin/env python3

import rospy

# To deal with conversions
import tf_conversions

# Allows to publish transforms
import tf2_ros

from geometry_msgs.msg import TransformStamped

if __name__ == "__main__":
    rospy.init_node("static_tf2_world_broadcaster")
    static_broadcaster = tf2_ros.StaticTransformBroadcaster()
    static_transformStamped = TransformStamped()

    static_transformStamped.header.stamp = rospy.Time.now()
    
    static_transformStamped.header.frame_id = "mobile_platform/world"
    static_transformStamped.child_frame_id = "camera1/world"
    
    static_transformStamped.transform.translation.x = 0.
    static_transformStamped.transform.translation.y = 0.
    static_transformStamped.transform.translation.z = 0.

    # quat = tf_conversions.transformations.quaternion_from_euler(0., 0., 0.)
    static_transformStamped.transform.rotation.x = 0.
    static_transformStamped.transform.rotation.y = 0.
    static_transformStamped.transform.rotation.z = 0.
    static_transformStamped.transform.rotation.w = 1.
    # static_transformStamped.transform.rotation.x = quat[0]
    # static_transformStamped.transform.rotation.y = quat[1]
    # static_transformStamped.transform.rotation.z = quat[2]
    # static_transformStamped.transform.rotation.w = quat[3]
    
    static_broadcaster.sendTransform(static_transformStamped)
    
    rospy.spin()
