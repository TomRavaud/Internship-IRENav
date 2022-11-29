import numpy as np


# Set a constant time step
dt = 0.1

# Compute the (constant) evolution matrix A
A = np.zeros((36, 36))

# The first six lines of A
A[:6, :6] = np.eye(6)
A[:6, 6:12] = dt*np.eye(6)
A[:6, 12:18] = dt**2/2*np.eye(6)
A[:6, 18:24] = dt**3/6*np.eye(6)
A[:6, 24:30] = dt**4/24*np.eye(6)
A[:6, 30:36] = dt**5/120*np.eye(6)

# The other lines
A[6:12, :] = np.roll(A[:6, :], 6)
A[12:18, :] = np.roll(A[:6, :], 12)
A[18:24, :] = np.roll(A[:6, :], 18)
A[24:30, :] = np.roll(A[:6, :], 24)
A[30:36, :] = np.roll(A[:6, :], 30)
    
# Save it in a file
np.save("A.npy", A)

# Set the (constant) observation matrix C
C = np.zeros((12, 36))
C[:6, :6] = np.eye(6)
C[6:12, :] = np.roll(C[0:6, :], 30)

# Save it in a file
np.save("C.npy", C)

# alpha and beta are supposed to be white Gaussian signals
Galpha = np.eye(36)
Gbeta = 10**(-3)*np.eye(12)  # Pose estimation is precise


# Set a number of iterations
k = 1000

# Initialize the covariance matrix G
G = 1e4 * np.eye(36)

# Solve iteratively the equation of Ricatti in G
for i in range(1000):
    G = A@(G - G@np.transpose(C)@np.linalg.inv(C@G@np.transpose(C) + Gbeta)@C@G)@np.transpose(A) + Galpha


# Compute the gain matrix to use in the stationary Kalman filter
K = G@np.transpose(C)@np.linalg.inv(C@G@np.transpose(C) + Gbeta)

# Save it in a file
np.save("K.npy", K)
