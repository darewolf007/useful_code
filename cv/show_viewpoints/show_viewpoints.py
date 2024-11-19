import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import math
def cartesian_to_spherical(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:, 0]**2 + xyz[:, 1]**2
    z = np.sqrt(xy + xyz[:, 2]**2)
    theta = np.arctan2(np.sqrt(xy), xyz[:, 2])
    azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
    return np.array([theta, azimuth, z])

def get_T(target_RT, cond_RT):
    R, T = target_RT[:3, :3], target_RT[:3, 3]
    T_target = -R.T @ T

    R, T = cond_RT[:3, :3], cond_RT[:3, 3]
    T_cond = -R.T @ T

    theta_cond, azimuth_cond, z_cond = cartesian_to_spherical(T_cond[None, :])
    theta_target, azimuth_target, z_target = cartesian_to_spherical(T_target[None, :])

    d_theta = theta_target - theta_cond
    d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
    d_z = z_target - z_cond

    d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
    return d_T, T_target, T_cond

data_delta_pose = np.load('/home/haowen/hw_useful_code/test_data/viewpoints/camera_poses.npy')
data_base_pose =  np.load('/home/haowen/hw_useful_code/test_data/viewpoints/cam0_wrt_table.npy')
view_points = np.copy(data_delta_pose)
for i in range(data_delta_pose.shape[0] ):
    view_points[i,:]=np.dot(data_base_pose,data_delta_pose[i,:])

print(get_T(data_delta_pose[1], data_delta_pose[3]))
print(get_T(view_points[0], view_points[3]))
viewpoints = view_points[:, :3, 3]

rotation_matrices = view_points[:, :3, :3]

fig = plt.figure(figsize=(12, 7))

ax3d = fig.add_subplot(121, projection='3d')

for i in range(viewpoints.shape[0]):
    x, y, z = viewpoints[i]

    R = rotation_matrices[i]
    camera_direction = R[:, 2]
    ax3d.quiver(x, y, z, camera_direction[0], camera_direction[1], camera_direction[2], 
                length=0.1, color='b', arrow_length_ratio=0.1)
    ax3d.scatter(x, y, z, c='r', marker='o')
ax3d.scatter(view_points[0, 0, 3], view_points[0, 1, 3], view_points[0, 2, 3], c='g', marker='o')
# ax3d.scatter(data_base_pose[0, 3], data_base_pose[1, 3], data_base_pose[2, 3], c='g', marker='o')

ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('Z')
ax3d.set_title('3D Viewpoints and Camera Direction')

ax2d = fig.add_subplot(122)
ax2d.scatter(viewpoints[:, 0], viewpoints[:, 1], c='r', marker='o')
ax2d.set_xlabel('X')
ax2d.set_ylabel('Y')
ax2d.set_title('2D Projection (XY Plane)')

plt.tight_layout()
plt.show()