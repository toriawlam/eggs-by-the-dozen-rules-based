# Standard imports
import cv2
import numpy as np;

kernel = np.ones((2, 2), np.uint8) 

# Read image
img = cv2.imread("strongylid_test_2.jpg", cv2.IMREAD_GRAYSCALE)
# gray= cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
im = cv2.erode(img, kernel, iterations=1) 

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 100
params.maxThreshold = 200

# Filter by Color
params.filterByColor = True

# Filter by Area.
# Change based on sample size/magnification
params.filterByArea = True
params.minArea = 3000
params.maxArea = 10000

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.05 #.01
params.maxInertiaRatio = 1

params.minDistBetweenBlobs = 0.00001

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)