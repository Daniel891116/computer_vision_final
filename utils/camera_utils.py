import numpy as np
from typing import List
import math

def quaternion_rotation_matrix(Q: List):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # # Extract the values from Q
    # q0 = Q[0]
    # q1 = Q[1]
    # q2 = Q[2]
    # q3 = Q[3]
     
    # # First row of the rotation matrix
    # r00 = 2 * (q0 * q0 + q1 * q1) - 1
    # r01 = 2 * (q1 * q2 - q0 * q3)
    # r02 = 2 * (q1 * q3 + q0 * q2)
     
    # # Second row of the rotation matrix
    # r10 = 2 * (q1 * q2 + q0 * q3)
    # r11 = 2 * (q0 * q0 + q2 * q2) - 1
    # r12 = 2 * (q2 * q3 - q0 * q1)
     
    # # Third row of the rotation matrix
    # r20 = 2 * (q1 * q3 - q0 * q2)
    # r21 = 2 * (q2 * q3 + q0 * q1)
    # r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    # rot_matrix = np.array([[r00, r01, r02],
    #                        [r10, r11, r12],
    #                        [r20, r21, r22]])
    rot_matrix = np.array(
        [[1.0 - 2 * (Q[1] * Q[1] + Q[2] * Q[2]), 2 * (Q[0] * Q[1] - Q[3] * Q[2]), 2 * (Q[3] * Q[1] + Q[0] * Q[2])],
         [2 * (Q[0] * Q[1] + Q[3] * Q[2]), 1.0 - 2 * (Q[0] * Q[0] + Q[2] * Q[2]), 2 * (Q[1] * Q[2] - Q[3] * Q[0])],
         [2 * (Q[0] * Q[2] - Q[3] * Q[1]), 2 * (Q[1] * Q[2] + Q[3] * Q[0]), 1.0 - 2 * (Q[0] * Q[0] + Q[1] * Q[1])]],
        dtype=np.float32)
                
    return rot_matrix

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians