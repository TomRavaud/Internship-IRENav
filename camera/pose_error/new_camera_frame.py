#!/usr/bin/env python3

import rospy
import tf2_msgs.msg
import geometry_msgs.msg
import tf_conversions
import numpy as np


class FixedTFBroadcaster:

    def __init__(self):
        self.pub_tf = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=1)

        while not rospy.is_shutdown():
            # Run this loop at about 10Hz
            rospy.sleep(0.1)

            t = geometry_msgs.msg.TransformStamped()
            t.header.frame_id = "camera1/camera_link"
            t.header.stamp = rospy.Time.now()
            t.child_frame_id = "camera1/camera_frame_oriented"
            t.transform.translation.x = 0.0
            t.transform.translation.y = 0.0
            t.transform.translation.z = 0.0
            
            quat = tf_conversions.transformations.quaternion_from_euler(np.pi/2, 0., np.pi/2)

            t.transform.rotation.x = quat[0]
            t.transform.rotation.y = quat[1]
            t.transform.rotation.z = quat[2]
            t.transform.rotation.w = quat[3]

            tfm = tf2_msgs.msg.TFMessage([t])
            self.pub_tf.publish(tfm)

if __name__ == '__main__':
    rospy.init_node('camera_frame_broadcaster')
    tfb = FixedTFBroadcaster()

    rospy.spin()
 