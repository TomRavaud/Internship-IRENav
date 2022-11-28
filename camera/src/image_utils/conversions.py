from geometry_msgs.msg import TransformStamped
import rospy
import tf.transformations
import numpy as np

def transform_matrix_to_tf(HTM):
     # Instantiate a TransformStamped message
    tf_msg = TransformStamped()
    
    # Fill the stamp, parent and child's id attributes
    tf_msg.header.stamp = rospy.Time.now()
    tf_msg.header.frame_id = "camera1/camera_frame_oriented"
    tf_msg.child_frame_id = "mobile_platform/board_upper_side"
    
    # Set the translation
    tf_msg.transform.translation.x = HTM[0, 3]
    tf_msg.transform.translation.y = HTM[1, 3]
    tf_msg.transform.translation.z = HTM[2, 3]
    
    # Convert the rotation matrix to quaternion
    q = tf.transformations.quaternion_from_matrix(HTM)
    # Set the rotation
    tf_msg.transform.rotation.x = q[0]
    tf_msg.transform.rotation.y = q[1]
    tf_msg.transform.rotation.z = q[2]
    tf_msg.transform.rotation.w = q[3]
    
    return tf_msg

def tf_to_transform_matrix(tf_msg):
    """Convert a Transform object to a numpy transform matrix

    Args:
        tf_msg (Transform): a ROS Transform message which contains
        a translation vector and a rotation matrix

    Returns:
        ndarray (4, 4): the corresponding transform matrix
    """
    # Make the translation vector a numpy array
    T = np.array([tf_msg.translation.x,
                  tf_msg.translation.y,
                  tf_msg.translation.z])
    
    # Make the quaternion a numpy array
    q = np.array([tf_msg.rotation.x,
                  tf_msg.rotation.y,
                  tf_msg.rotation.z,
                  tf_msg.rotation.w])
    
    # Form the transform matrix from the translation and the quaternion
    HTM = tf.transformations.quaternion_matrix(q)
    HTM[0:3, 3] = T
    
    return HTM
