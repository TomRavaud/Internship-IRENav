import numpy as np
import cv2
import itertools


def ransac(old_points, points, nb_iterations):
    random_id = np.random.sample(range(len(points)), k=4)
    
    print(random_id)
    
    random_old_points = old_points[random_id]
    random_points = points[random_id]
    
    # Compute homography
    
    # Compute inliers (all points)
    
    
L = range(40)

c = list(itertools.combinations(L, 4))

print(c[0])