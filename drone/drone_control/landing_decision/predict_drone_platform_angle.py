import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("/media/tom/Shared/Stage-EN-2022/quadcopter_landing_ws/src/drone/drone_control/landing_decision/record_drone_platform_roll_20_01.csv",
                   sep=",",
                   decimal=".",
                   index_col=0)

drone_platform_angle = data['Angle']

linear_motion_matrix_1D = np.array([[2, -1],
                                    [1, 0]])

quadratic_motion_matrix_1D = np.array([[3, -3, 1],
                                       [1, 0, 0],
                                       [0, 1, 0]])

angle_predict_linear = []
angle_predict_quadratic = []

for i in range(100, 200):
    s_2 = np.array([drone_platform_angle[i-1], drone_platform_angle[i-2]])
    s_predict_linear = np.dot(linear_motion_matrix_1D, s_2)
    angle_predict_linear.append(s_predict_linear[0])
    
    s_3 = np.array([drone_platform_angle[i-1], drone_platform_angle[i-2], drone_platform_angle[i-3]])
    s_predict_quadratic = np.dot(quadratic_motion_matrix_1D, s_3)
    angle_predict_quadratic.append(s_predict_quadratic[0])

plt.figure()
plt.title("Predicted roll angle")

plt.subplot("121")
plt.plot(drone_platform_angle[100:200], "ro")
plt.plot(range(100, 200), angle_predict_linear, "bo")
plt.title("Linear motion")
plt.xlabel("Timestamp")
plt.ylabel("Roll angle (rad)")

plt.subplot("122")
plt.plot(drone_platform_angle[100:200], "ro")
plt.plot(range(100, 200), angle_predict_quadratic, "bo")
plt.title("Quadratic motion")
plt.xlabel("Timestamp")
plt.ylabel("Roll angle (rad)")

plt.show()
