import numpy as np
import cv2

from image_utils import drawing as dw


# Load the gray-scaled reference image in which we know
# the position of the template
REFERENCE = cv2.imread(
    "./camera/src/correspondences_2d3d/Images/reference_image.png",
    flags=cv2.IMREAD_GRAYSCALE)

# Height and width of the reference frame
Rh, Rw = np.shape(REFERENCE)

# Max displacement of each corner point along x and y axes
# We allow here a maximum displacement of 2.5% of the image's lowest dimension
# DMU_MAX = 0.025*np.min([Rh, Rw])
DMU_MAX = 10

# The four 2D corner points of the template region in the reference frame
x1, x2 = 239, 561
y1, y2 = 239, 561

TEMPLATE_CORNERS_REF = np.array([[x2, y2],
                                 [x1, y2],
                                 [x1, y1],
                                 [x2, y1]])

# Reshape the corners array
TEMPLATE_CORNERS_REF = TEMPLATE_CORNERS_REF.reshape(8, 1)

# Sample the image to a reduced grid of points
NB_POINTS_1D = 10
NB_POINTS_2D = NB_POINTS_1D**2

X = np.linspace(x1, x2, NB_POINTS_1D, dtype=np.int32)
Y = np.linspace(y1, y2, NB_POINTS_1D, dtype=np.int32)

NB_POINTS_2D = 36

# Grid inside the template
X_grid, Y_grid = np.meshgrid(X, Y)
X_grid = X_grid[2:-2, 2:-2]
Y_grid = Y_grid[2:-2, 2:-2]

X_grid_flat = X_grid.ravel()
Y_grid_flat = Y_grid.ravel()

# print(X_grid, Y_grid)

# print(X_grid_flat)

points = np.zeros((NB_POINTS_2D, 2))

for k in range(NB_POINTS_2D):
    points[k, 0] = X_grid_flat[k]
    points[k, 1] = Y_grid_flat[k]
    
# print(tuple(points[0]))
# print(points)

# REFERENCE = dw.draw_points(REFERENCE, points)

# cv2.imshow("Reference frame", REFERENCE)
# cv2.waitKey()

# Get the intensity of the points on the grid and store them in a column array
I_REF = np.int32(REFERENCE[X_grid, Y_grid].reshape(NB_POINTS_2D, 1))
# print(I_REF)
# print(REFERENCE[525, 525])

# Number of reference image's transformations in the training set
NB_TRANSFORMATIONS = 100

# assert NB_POINTS_2D < NB_TRANSFORMATIONS, "The number of transformations\
# must be much larger than the number of points"

# Matrix containing the vector of small disturbances to the template position
# parameters
Y = np.zeros((8, NB_TRANSFORMATIONS))

# Matrix storing image intensity differences
H = np.zeros((NB_POINTS_2D, NB_TRANSFORMATIONS))

for i in range(NB_TRANSFORMATIONS):
    # Compute a random small displacement of the corners
    # dmu_i = np.random.randint(-DMU_MAX, DMU_MAX, size=(8, 1))
    d = np.random.randint(-DMU_MAX, DMU_MAX)
    dmu_i = np.array([[0],
                      [d],
                      [0],
                      [d],
                      [0],
                      [d],
                      [0],
                      [d]])
    
    # dmu_i = np.random.rand(8,1)*2*DMU_MAX - DMU_MAX
    
    # Fill the corresponding column in the small disturbances matrix
    Y[:, i] = dmu_i[:, 0]
    
    # Shift reference template corners' coordinates
    new_template_corners = TEMPLATE_CORNERS_REF + dmu_i
    
    new_template_corners = new_template_corners.reshape(4, 2)
    
    #TODO: Put it out of the for loop
    TEMPLATE_reshape = TEMPLATE_CORNERS_REF.reshape(4, 2)
    
    new_template_corners = np.float32(new_template_corners)
    TEMPLATE_reshape = np.float32(TEMPLATE_reshape)
    # print(TEMPLATE_reshape.shape)
    # print(new_template_corners.shape)
    
    # Set the warping function to be a homography (= perspective transform)
    F = cv2.getPerspectiveTransform(TEMPLATE_reshape, new_template_corners)
    # print(F)
    # F = cv2.findHomography(TEMPLATE_CORNERS_REF, new_template_corners)
    
    #TODO: Compute the new meshgrid
    # a = np.array([[2, 3]])
    # tranfo = cv2.perspectiveTransform(a, F)
    # print(a)
    # new_X_grid, new_Y_grid = cv2.perspectiveTransform(X_grid, Y_grid, F)
    new_image = cv2.warpPerspective(REFERENCE, F, (Rw, Rh))
    
    # ref_copy = np.copy(new_image)
    # ref_copy = dw.draw_points(ref_copy, new_template_corners)
    # ref_copy = dw.draw_points(ref_copy, points)
    
    # cv2.imshow(str(i), ref_copy)
    # cv2.waitKey()
    
    # Compute the differences in intensities
    # print(i)
    # print(I_REF)
    # print(np.int32(new_image[X_grid, Y_grid]))
    di_i = np.int32(new_image[X_grid, Y_grid].reshape(NB_POINTS_2D, 1)) - I_REF
    # print(np.shape(di_i))
    # print(di_i)
    # print(I_REF)
    # print(new_image[X_grid, Y_grid].reshape(NB_POINTS_2D, 1))
    # print(i)
    H[:, i] = di_i[:, 0]
    # print(di_i[-1, 0])
    
# print(np.shape(Y))
# print(np.shape(H))

# TODO: Other way to compute a right pseudo inverse
# print(np.sum(H, 1))
# print(H)
# print(np.sum(np.dot(H, np.transpose(H)), 0))

err = np.random.random((NB_POINTS_2D, NB_TRANSFORMATIONS))/10
H += err
H_pinv = np.dot(np.transpose(H), np.linalg.inv(np.dot(H, np.transpose(H))))
# print(np.dot(H, np.transpose(H)))

# U, D, Vt = np.linalg.svd(H)
# print(np.shape(U))
# print(np.shape(D))
# print(np.shape(Vt))

A = np.dot(Y, H_pinv)
print(A)

# Save the off-line pre-computed matrix A in a file so that we can load it
# easily in other scripts
np.save("A.npy", A)
