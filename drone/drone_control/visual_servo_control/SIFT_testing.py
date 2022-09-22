import cv2

# print(cv2.__version__)
im1 = cv2.imread("/media/tom/Shared/Stage-EN-2022/quadcopter_landing_ws/src/drone/drone_control/visual_servo_control/target_image_1475.png", flags=cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread("/media/tom/Shared/Stage-EN-2022/quadcopter_landing_ws/src/drone/drone_control/visual_servo_control/drone_image_scale_rotation.png", cv2.IMREAD_GRAYSCALE)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Find the keypoints and their descriptor with SIFT
kp1, des1 = sift.detectAndCompute(im1, None)
kp2, des2 = sift.detectAndCompute(im2, None)

# Brute-force matching with L2 norm and cross check to ensure better results
bf = cv2.BFMatcher.create(normType=cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches
matches = sorted(matches, key = lambda x:x.distance)

# print(matches[0].queryIdx, matches[0].trainIdx)
# print(matches[:10])
# print(kp1[0].pt)

# Draw matches
im_matches = cv2.drawMatches(im1, kp1, im2, kp2, matches[:10], None, flags=
                             cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# cv2.imwrite("keypoints_matching.png", im_matches)

# print(kp1)
# print(cv2.KeyPoint.convert(kp1))
# matches = matches[:10]
# print(matches)

cv2.imshow('image', im_matches)
cv2.waitKey()
