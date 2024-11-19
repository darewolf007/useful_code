import numpy as np
def euler_to_quaternion(euler, degree_mode=1):
    roll, yaw, pitch = euler
    if degree_mode == 1:
        roll = np.deg2rad(roll)
        pitch = np.deg2rad(pitch)
        yaw = np.deg2rad(yaw)

    qx = np.sin(pitch/2) * np.cos(yaw/2) * np.cos(roll/2) - np.cos(pitch/2) * np.sin(yaw/2) * np.sin(roll/2)
    qy = np.cos(pitch/2) * np.sin(yaw/2) * np.cos(roll/2) + np.sin(pitch/2) * np.cos(yaw/2) * np.sin(roll/2)
    qz = np.cos(pitch/2) * np.cos(yaw/2) * np.sin(roll/2) - np.sin(pitch/2) * np.sin(yaw/2) * np.cos(roll/2)
    qw = np.cos(pitch/2) * np.cos(yaw/2) * np.cos(roll/2) + np.sin(pitch/2) * np.sin(yaw/2) * np.sin(roll/2)
    q = np.array([qx, qy, qz, qw])
    return q