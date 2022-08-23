import numpy as np
import cv2
import tf.transformations


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

def error_function(HTM1, HTM2):
    """Compute an error between two transform matrices supposed to be equal

    Args:
        HTM1 (ndarray (4, 4)): the first transform matrix
        HTM2 (ndarray (4, 4)): the second transform matrix

    Returns:
        tuple (2): errors on the translation and on the rotation
    """
    R1 = HTM1[:3, :3]
    R2 = HTM2[:3, :3]
    
    # Compute the transformation matrix between 1 and 2
    HTM12 = np.dot(np.linalg.inv(HTM1), HTM2)
    
    # Extract the translation vector and the rotation matrix
    T12 = HTM12[:3, 3]
    R12 = HTM12[:3, :3]
    
    # Compute the error (in meters) on the translation
    error_translation = np.linalg.norm(T12)
    
    # Compute the error (in radians) on the rotation
    # (we apply the Euclidian norm to rotation vector converted from the
    # rotation matrix using the Rodrigues' formula)
    error_rotation = np.linalg.norm(cv2.Rodrigues(R12)[0])
    
    return error_translation, error_rotation
