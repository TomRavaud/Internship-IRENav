import numpy as np
import cv2

from image_processing import drawing as dw

# Load the gray-scaled reference image in which we know
# the position of the template
REFERENCE = cv2.imread(
    "image.png",
    flags=cv2.IMREAD_GRAYSCALE)

# Height and width of the reference frame
Rh, Rw = np.shape(REFERENCE)

# Max displacement of each corner point along x and y axes
# We allow here a maximum displacement of 2.5% of the image's lowest dimension
# DMU_MAX = 0.025*np.min([Rh, Rw])
DMU_MAX = 10

# The four 2D corner points of the template region in the image reference
x1_i, x2_i = 239, 561
y1_i, y2_i = 239, 561

mu = np.array([[x1_i, y1_i],
               [x2_i, y1_i],
               [x2_i, y2_i],
               [x1_i, y2_i]])

# In the reference region
x1_r, x2_r = 0, 322
y1_r, y2_r = 0, 322

MU = np.array([[x1_r, y1_r],
               [x2_r, y1_r],
               [x2_r, y2_r],
               [x1_r, y2_r]])

# First homography
F0, _ = cv2.findHomography(MU, mu)

# Sample the image to a reduced grid of points
NB_POINTS_1D = 10

X = np.linspace(x1_r, x2_r, NB_POINTS_1D)
Y = np.linspace(y1_r, y2_r, NB_POINTS_1D)

NB_POINTS_2D = 64 # (10-2)**2

# Grid inside the template in the region reference
X_grid_r, Y_grid_r = np.meshgrid(X, Y)
X_grid_r = X_grid_r[1:-1, 1:-1]
Y_grid_r = Y_grid_r[1:-1, 1:-1]

# X_grid_flat = X_grid.ravel()
# Y_grid_flat = Y_grid.ravel()

# points = np.zeros((NB_POINTS_2D, 2))

# for k in range(NB_POINTS_2D):
#     points[k, 0] = X_grid_flat[k]
#     points[k, 1] = Y_grid_flat[k]

def grid_homography(X_grid, Y_grid, F):
    # Update grid points coordinates
    f11, f12, f13 = F[0]
    f21, f22, f23 = F[1]
    f31, f32, f33 = F[2]

    X_grid_new = np.int32((f11*X_grid + f12*Y_grid + f13) / (f31*X_grid + f32*Y_grid + f33))
    Y_grid_new = np.int32((f21*X_grid + f22*Y_grid + f23) / (f31*X_grid + f32*Y_grid + f33))
    
    return X_grid_new, Y_grid_new
    
X_grid_i, Y_grid_i = grid_homography(X_grid_r, Y_grid_r, F0)

# Get the intensity of the points on the grid and store them in a column array
I_REF = np.float32(REFERENCE[X_grid_i, Y_grid_i].reshape(NB_POINTS_2D, 1))

# Number of reference image's transformations in the training set
NB_TRANSFORMATIONS = 10000

# Matrix containing the vector of small disturbances to the template position
# parameters
Y = np.zeros((8, NB_TRANSFORMATIONS))

# Matrix storing image intensity differences
H = np.zeros((NB_POINTS_2D, NB_TRANSFORMATIONS))


for i in range(NB_TRANSFORMATIONS):
    # Compute a random small displacement of the corners
    dmu = np.random.randint(-DMU_MAX, DMU_MAX, size=(4, 2))
    
    F10, _ = cv2.findHomography(mu + dmu, mu)
    F1 = np.dot(F0, F10)
    
    X_grid_i_new, Y_grid_i_new = grid_homography(X_grid_r, Y_grid_r, F1)
    
    i_current = np.float32(REFERENCE[X_grid_i_new, Y_grid_i_new].reshape(NB_POINTS_2D, 1))
    
    di = i_current - I_REF
    
    # Fill the corresponding column in the small disturbances matrix
    Y[:, i] = dmu.reshape(8, 1)[:, 0]
    H[:, i] = di[:, 0]
    
# err = np.random.random((NB_POINTS_2D, NB_TRANSFORMATIONS))/10
# H += err
H_pinv_right = np.dot(np.transpose(H), np.linalg.inv(np.dot(H, np.transpose(H))))
H_pinv_left = np.dot(np.linalg.inv(np.dot(np.transpose(H), H)), np.transpose(H))
# print(np.dot(H, np.transpose(H)))

A_right = np.dot(Y, H_pinv_right)
A_left = np.dot(Y, H_pinv_left)

# Save the off-line pre-computed matrix A in a file so that we can load it
# easily in other scripts
np.save("pre_computed_A_right.npy", A_right)
np.save("pre_computed_A_left.npy", A_left)
