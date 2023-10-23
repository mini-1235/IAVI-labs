import cv2 as cv
import glob
import numpy as np
import matplotlib.pyplot as plt
import PIL.ExifTags
import PIL.Image

def calibrate_camera(images_folder):
    images_names = glob.glob(images_folder)
    images = []
    for image_name in images_names:
        image = cv.imread(image_name)
        images.append(image)
    
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100000, 0.00001)

    # Chessboard size
    rows = 6
    columns = 8

    # frame size
    width = images[0].shape[1]
    height = images[0].shape[0]

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns, 3), np.float32)
    objp[:,:2] = np.mgrid[0:rows, 0:columns].T.reshape(-1,2)

    imgpoints = [] # 2D points in image plane
    objpoints = [] # 3D points in real world space

    for frame in images:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)
        if ret == True:
            corners = cv.cornerSubPix(gray, corners, (20, 20), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners)
            cv.drawChessboardCorners(frame, (rows, columns), corners, ret)
    
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print(ret)
    return mtx, dist, rvecs, tvecs

def stereo_calibrate(mtx1, dist1, mtx2, dist2, images_folder1, images_folder2):
    # read the images
    c1_images_names = glob.glob(images_folder1)
    c2_images_names = glob.glob(images_folder2)

    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        c1_images.append(cv.imread(im1))
        c2_images.append(cv.imread(im2))
    
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100000, 0.00001)

    # Chessboard size
    rows = 6
    columns = 8

    # frame size
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns, 3), np.float32)
    objp[:,:2] = np.mgrid[0:rows, 0:columns].T.reshape(-1,2)

    imgpoints_left = [] # 2D points in image plane
    imgpoints_right = [] # 2D points in image plane

    objpoints = [] # 3D points in real world space

    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
        ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)

        if ret1 == True and ret2 == True:
            corners1 = cv.cornerSubPix(gray1, corners1, (20, 20), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (20, 20), (-1, -1), criteria)
            cv.drawChessboardCorners(frame1, (rows, columns), corners1, ret1)
            cv.drawChessboardCorners(frame2, (rows, columns), corners2, ret2)
            
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
            objpoints.append(objp)

    sterocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dis2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1, mtx2, dist2, (width, height), flags=sterocalibration_flags)

    print(ret)
    return ret, CM1, dist1, CM2, dis2, R, T, E, F


mtx1, dist1, rvecs1, tvecs1 = calibrate_camera("left_2/*")
mtx2, dist2, rvecs2, tvecs2 = calibrate_camera("right_2/*")
print("f1: " + str(mtx1[0][0]))
print("f2: " + str(mtx2[0][0]))

ret, CM1, dist1, CM2, dis2, R, T, E, F = stereo_calibrate(mtx1, dist1, mtx2, dist2, "lcheck2/l1.bmp", "rcheck2/r1.bmp")


# read the images
left_images = cv.imread("l10.bmp")
right_images = cv.imread("r10.bmp")

# undistort the images
left_undistorted =  cv.undistort(left_images, mtx1, dist1)
right_undistorted = cv.undistort(right_images, mtx2, dist2)
# cv.imshow("left_undistorted", left_undistorted)
# cv.imshow("right_undistorted", right_undistorted)
# cv.waitKey(0)

# Rectify the images
width = left_images.shape[1]
height = left_images.shape[0]
R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(mtx1, dist1, mtx2, dist2, (width, height), R, T)
left_map1, left_map2 = cv.initUndistortRectifyMap(mtx1, dist1, R1, P1, (width, height), cv.CV_32FC1)
right_map1, right_map2 = cv.initUndistortRectifyMap(mtx2, dist2, R2, P2, (width, height), cv.CV_32FC1)
left_rectified = cv.remap(left_undistorted, left_map1, left_map2, cv.INTER_LINEAR)
right_rectified = cv.remap(right_undistorted, right_map1, right_map2, cv.INTER_LINEAR)
# cv.imshow("left_rectified", left_rectified)
# cv.imshow("right_rectified", right_rectified)
# cv.waitKey(0)
# cv.destroyAllWindows()

# compute the disparity map
win_size = 3
min_disp = -1 - 16*0
max_disp = 15 + 16*10
num_disp = max_disp - min_disp # Needs to be divisible by 16
stereo = cv.StereoSGBM_create(
        minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = win_size,
        uniquenessRatio = 1,
        speckleWindowSize = 20,
        speckleRange = 50,
        disp12MaxDiff = 2,
        P1 = 16 * win_size**2,
        P2 = 256 * win_size**2,
        preFilterCap = 100,
        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
    )

# use WLS filter
left_matcher = stereo
right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
# FILTER Parameters
lmbda = 80000
sigma = 1.3
visual_multiplier = 6

wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

displ = left_matcher.compute(left_rectified, right_rectified)
dispr = right_matcher.compute(right_rectified, left_rectified)
displ = np.int16(displ) * 8
dispr = np.int16(dispr) * 8
filteredImg = wls_filter.filter(displ, left_rectified, None, dispr)

filteredImg = cv.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv.NORM_MINMAX)
filteredImg = np.uint8(filteredImg)

plt.imshow(dispr)
plt.show()
plt.imshow(filteredImg)
plt.show()

# generate point cloud
h, w = left_rectified.shape[:2]

# Compute the focal length from the calibration matrix
focal_length = mtx1[0][0] / 500

Q = np.float32([[1, 0, 0, 0],
                [0,-1, 0, 0],
                [0, 0, focal_length*0.05, 0], # Focal length multiplication obtained experimentally. 
                [0, 0, 0, 1]])

# Reproject points into 3D
points_3D = cv.reprojectImageTo3D(dispr, Q)
# Get color points
colors = cv.cvtColor(left_rectified, cv.COLOR_BGR2RGB)

# Get rid of points with value 0 (i.e no depth)
mask_map = dispr > dispr.min()

# Mask colors and points.
output_points = points_3D[mask_map]
output_colors = colors[mask_map]

# Define name for output file
output_file = 'reconstructed.ply'
print ("\nSaving output file\n")
with open(output_file, 'w') as f:
    # Header
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write("element vertex %d\n" % len(output_points))
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("property uchar red\n")
    f.write("property uchar green\n")
    f.write("property uchar blue\n")
    f.write("end_header\n")

    # Pointcloud
    for i in range(len(output_points)):
        f.write("%f %f %f %d %d %d\n" % (output_points[i][0], output_points[i][1], output_points[i][2], output_colors[i][0], output_colors[i][1], output_colors[i][2]))

# Reproject points into 3D
points_3D = cv.reprojectImageTo3D(filteredImg, Q)
# Get color points
colors = cv.cvtColor(left_rectified, cv.COLOR_BGR2RGB)

# Get rid of points with value 0 (i.e no depth)
mask_map = filteredImg > filteredImg.min()

# Mask colors and points.
output_points = points_3D[mask_map]
output_colors = colors[mask_map]

# Define name for output file
output_file = 'reconstructed2.ply'
print ("\nSaving output file\n")
with open(output_file, 'w') as f:
    # Header
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write("element vertex %d\n" % len(output_points))
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("property uchar red\n")
    f.write("property uchar green\n")
    f.write("property uchar blue\n")
    f.write("end_header\n")

    # Pointcloud
    for i in range(len(output_points)):
        f.write("%f %f %f %d %d %d\n" % (output_points[i][0], output_points[i][1], output_points[i][2], output_colors[i][0], output_colors[i][1], output_colors[i][2]))
