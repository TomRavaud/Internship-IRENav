import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True  # Render Matplotlib text with Tex

import cv2

def apply_homography(src, H):
    return (cv2.perspectiveTransform(src.reshape(-1, 1, 2).astype(np.float32), H)).reshape(-1, 2)

# corners_image_2D = np.array([[561, 561],
#                                      [239, 561],
#                                      [239, 239],
#                                      [561, 239]], dtype="double")
# CORNERS_PLATFORM_2D = np.array([[-0.5, -0.5],
#                                 [-0.5, 0.5],
#                                 [0.5, 0.5],
#                                 [0.5, -0.5]])
# # The homography matrix is defined up to a scale factor
# H, _ = cv2.findHomography(corners_image_2D, CORNERS_PLATFORM_2D)

# x1, x2 = 300, 450
# y1, y2 = 300, 450
# # x1_i, x2_i = 239, 561
# # y1_i, y2_i = 239, 561

# mu0 = np.array([[x1, y1],
#                [x2, y1],
#                [x2, y2],
#                [x1, y2]], dtype=np.float32)

# MU_PLATFORM = apply_homography(mu0, H)
# print(MU_PLATFORM)

T = np.load("T.npy")
print(np.mean(T/1000))

MU = np.load("mu.npy")
# print(np.shape(MU))

MU_true = np.load("mu_true.npy")

# error = MU - MU_true
# print(error)
# print(MU_true)

# error = np.zeros_like(MU)
# error = np.linalg.norm(MU - MU_true, axis = 0)

# plt.figure()
# plt.plot(error, "r.")
# plt.plot(MU[:, 7], "r.")
# plt.plot(MU_true[:, 7], "b.")
# plt.plot(range(1, 1001), T, "r.")
# plt.title(r"Tracking time")
# plt.xlabel(r"Step $k$")
# plt.ylabel(r"Tracking time (in ms)")
# plt.show()
