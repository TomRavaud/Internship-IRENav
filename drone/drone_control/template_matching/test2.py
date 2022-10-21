import numpy as np
import cv2

from image_processing import drawing as dw


def grid_homography(X_grid, Y_grid, F):
    # Update grid points coordinates
    f11, f12, f13 = F[0]
    f21, f22, f23 = F[1]
    f31, f32, f33 = F[2]

    X_grid_new = np.int32((f11*X_grid + f12*Y_grid + f13) / (f31*X_grid + f32*Y_grid + f33))
    Y_grid_new = np.int32((f21*X_grid + f22*Y_grid + f23) / (f31*X_grid + f32*Y_grid + f33))
    
    return X_grid_new, Y_grid_new
 

# Load the gray-scaled reference image in which we know
# the position of the template
REF = cv2.imread(
    "image.png")

REFERENCE = cv2.cvtColor(REF, cv2.COLOR_BGR2GRAY)

# Height and width of the reference frame
Rh, Rw = np.shape(REFERENCE)

# Number of level (coarse to fine manner)
NB_LEVEL = 5

# Max displacement of each corner point along x and y axes
# We allow here a maximum displacement of 2.5% of the image's lowest dimension
# DMU_MAX = 0.020*np.min([Rh, Rw])
DMU_MAX = 10
# DMU_MAX = np.linspace(25, 3, NB_LEVEL)

# The four 2D corner points of the template region in the image reference
x1, x2 = 300, 450
y1, y2 = 300, 450
# x1_i, x2_i = 239, 561
# y1_i, y2_i = 239, 561

mu0 = np.array([[x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]])

# In the reference region
# x1_r, x2_r = 0, np.abs(x1_i - x2_i)
# y1_r, y2_r = 0, np.abs(y1_i - y2_i)

# MU = np.array([[x1_r, y1_r],
#                [x2_r, y1_r],
#                [x2_r, y2_r],
#                [x1_r, y2_r]])

# Sample the image into a reduced grid of points
NB_POINTS_1D = 30

X = np.linspace(x1, x2, NB_POINTS_1D, dtype=np.int32)
Y = np.linspace(y1, y2, NB_POINTS_1D, dtype=np.int32)

NB_POINTS_2D = NB_POINTS_1D**2

# Grid inside the template in the region reference
X_grid0, Y_grid0 = np.meshgrid(X, Y)

# X_grid_i, Y_grid_i = grid_homography(X_grid_r, Y_grid_r, F0)

# Get the intensity of the points on the grid and store them in a column array
I_REF = np.float32(REFERENCE[X_grid0, Y_grid0].reshape(NB_POINTS_2D, 1))

# Number of reference image's transformations in the training set
NB_TRANSFORMATIONS = 5000

A = np.zeros((8, NB_POINTS_2D))
# A = np.zeros((8, NB_POINTS_2D*NB_LEVEL))

# for l in range(NB_LEVEL):
# Matrix containing the vector of small disturbances to the template position
# parameters
Y = np.zeros((8, NB_TRANSFORMATIONS))
# Matrix storing image intensity differences
H = np.zeros((NB_POINTS_2D, NB_TRANSFORMATIONS))

for t in range(NB_TRANSFORMATIONS):
    
    # Compute a random small displacement of the corners
    dmu = np.random.randint(-DMU_MAX, DMU_MAX, size=(4, 2))
    # dmu = np.random.randint(-DMU_MAX[l], DMU_MAX[l], size=(4, 2))
    
    # Compute the homography between the reference template and
    # the transformed one
    F, _ = cv2.findHomography(mu0, mu0 + dmu)
    
    # Warp the grid with the transformation
    X_grid, Y_grid = grid_homography(X_grid0, Y_grid0, F)
    
    # Get intensities at the grid points
    i_current = np.float32(REFERENCE[X_grid, Y_grid].reshape(NB_POINTS_2D, 1))
    
    # Compute the differences with the reference intensities
    di = i_current - I_REF
    
    # Fill the corresponding column in the small disturbances matrix
    Y[:, t] = dmu.reshape(8, 1)[:, 0]
    H[:, t] = di[:, 0]
    #TODO: Normalize i and add random noise to H to avoid singularities
    
    # image_to_display = np.copy(REF)
    
    # X_grid_flat = X_grid.ravel()
    # Y_grid_flat = Y_grid.ravel()
    
    # points = np.zeros((NB_POINTS_2D, 2))
    
    # for k in range(NB_POINTS_2D):
    #     points[k, 0] = X_grid_flat[k]
    #     points[k, 1] = Y_grid_flat[k]
        
    # dw.draw_points(image_to_display, points)
    # dw.draw_points(image_to_display, mu0 + dmu, color=(0, 255, 0))
    
    # cv2.imshow(str(t), image_to_display)
    # cv2.waitKey()

# Least squares estimation of A
H_pinv = np.dot(np.transpose(H), np.linalg.inv(np.dot(H, np.transpose(H))))
# A[:, l*NB_POINTS_2D:(l+1)*NB_POINTS_2D] = np.dot(Y, H_pinv)
A = np.dot(Y, H_pinv)

# Save the off-line pre-computed matrix A in a file so that we can load it
# easily in other scripts
np.save("A.npy", A)
