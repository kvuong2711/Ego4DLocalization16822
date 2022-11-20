import numpy as np
from scipy.spatial.transform import Rotation as R_

np.random.seed(0)

# s = 5
# T = np.eye(4)
# R = np.eye(3)
# t = np.ones(3) * 2
# T[:3, :3] = s * R
# T[:3, 3] = t
#
# X1 = np.random.uniform(size=(3,))
#
# R1 = R_.random().as_matrix()
# t1 = np.random.uniform(size=(3,))
# P1 = np.eye(4)
# P1[:3, :3] = R1
# P1[:3, 3] = t1
#
# # print(X1.shape, P1.shape)
# # x_p = P1[:3, :3] @ X1 + P1[:3, 3]
#
# # print(x_p)
#
# P2_p = P1 @ np.linalg.inv(T)
# P2 = P2_p * np.linalg.norm(T[0, :3])
# R2 = P2[:3, :3]
# t2 = P2[:3, 3]
#
# print('sol R2:', R2)
# print('sol t2:', t2)
#
# X2 = s * R @ X1 + t
#
# # print(X1, X2)
# print(R2 @ X2 + t2)
#
# # R2 = 1/s * R1 @ T[:3, :3].T
# # t2 = -R2 @ T[:3, 3] + t1
# #
# # print(R2, t2)
# #
# # print(R2 @ X2 + t2)
# print(R1 @ X1 + t1)
#
# R2_est = R1 @ R.T
# t2_est = s * t1 - R2_est @ t
# print('cal R2:', R2_est)
# print('cal t2:', t2_est)

x = np.random.uniform(size=(3, 200))

x_homo = np.row_stack((x, np.ones(x.shape[1],)))
print(x_homo)



