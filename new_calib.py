import cv2 as cv
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure 
from skimage.exposure import match_histograms 
def radiometric_calibration(left_rectified, right_rectified):
    left_channels = cv.split(left_rectified)
    right_channels = cv.split(right_rectified)

    calibrated_right_channels = []
    for i in range(len(left_channels)):
        # Perform histogram equalization on each channel
        left_eq = cv.equalizeHist(left_channels[i])
        right_eq = cv.equalizeHist(right_channels[i])

        # Compute the CDFs (Cumulative Distribution Functions) of the channels
        left_cdf = np.cumsum(cv.calcHist([left_eq], [0], None, [256], [0, 256]))
        left_cdf_normalized = left_cdf / left_cdf[-1]

        right_cdf = np.cumsum(cv.calcHist([right_eq], [0], None, [256], [0, 256]))
        right_cdf_normalized = right_cdf / right_cdf[-1]

        # Find the matching values in the CDFs
        matched_right_cdf = np.interp(left_cdf_normalized, right_cdf_normalized, range(256))

        # Apply the matching values to the right channel
        calibrated_right_channel = np.interp(right_eq, range(256), matched_right_cdf).astype(np.uint8)
        calibrated_right_channels.append(calibrated_right_channel)

    # Merge the calibrated channels and convert back to BGR color space
    calibrated_right_bgr = cv.merge(calibrated_right_channels)

    return calibrated_right_bgr

def save_rectification_params(R1, R2, P1, P2, Q, roi1, roi2, filename):
    """ Save rectification parameters to a file """
    cv_file = cv.FileStorage(filename, cv.FILE_STORAGE_WRITE)
    cv_file.write("R1", R1)
    cv_file.write("R2", R2)
    cv_file.write("P1", P1)
    cv_file.write("P2", P2)
    cv_file.write("Q", Q)
    cv_file.write("roi1", roi1)
    cv_file.write("roi2", roi2)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()
def save_single_camera_params(mtx, dist, filename):
    """ Save camera parameters to a file """
    cv_file = cv.FileStorage(filename, cv.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()
def save_stereo_params(mtx1, dist1, mtx2, dist2, R, T, E, F,width,height,filename):
    """ Save stereo parameters to a file """
    cv_file = cv.FileStorage(filename, cv.FILE_STORAGE_WRITE)
    cv_file.write("K1", mtx1)
    cv_file.write("D1", dist1)
    cv_file.write("K2", mtx2)
    cv_file.write("D2", dist2)
    cv_file.write("R", R)
    cv_file.write("T", T)
    cv_file.write("E", E)
    cv_file.write("F", F)
    cv_file.write("width", width)
    cv_file.write("height", height)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()
def generate_point_cloud(left_rectified, right_rectified, Q, K1, D1, K2, D2, R1, R2, P1, P2, output_file):
    # Compute disparity map
    left_gray = cv.cvtColor(left_rectified, cv.COLOR_BGR2GRAY)
    right_gray = cv.cvtColor(right_rectified, cv.COLOR_BGR2GRAY)
    disparity_image = depth_map(left_gray, right_gray)

    # Apply radiometric calibration to right view
    radiometric_matrix = radiometric_calibration(left_rectified, right_rectified)
    right_rectified_calibrated = cv.transform(right_rectified, radiometric_matrix)

    # Generate 3D point cloud
    points_3d = cv.reprojectImageTo3D(disparity_image, Q)

    # Get color points
    colors_left = cv.cvtColor(left_rectified, cv.COLOR_BGR2RGB)
    colors_right = cv.cvtColor(right_rectified_calibrated, cv.COLOR_BGR2RGB)

    # Mask colors and points
    mask_map = disparity_image > disparity_image.min()
    output_points = points_3d[mask_map]
    output_colors = np.zeros_like(output_points)
    for i, point in enumerate(output_points):
        x, y, z = point
        object_points = np.array([[x, y, z]], dtype=np.float32)
        u, v = cv.fisheye.projectPoints(object_points, np.zeros((4, 1)), np.eye(3), None, K1)[0][0]
        if 0 <= u < colors_left.shape[1] and 0 <= v < colors_left.shape[0]:
            output_colors[i] = colors_left[int(v), int(u)]
        else:
            object_points = np.array([[x, y, z]], dtype=np.float32)
            u, v = cv.fisheye.projectPoints(object_points, np.zeros((4, 1)), R2, P2, K2)[0][0]
            if 0 <= u < colors_right.shape[1] and 0 <= v < colors_right.shape[0]:
                output_colors[i] = colors_right[int(v), int(u)]

    # Define output file format
    with open(output_file, 'w') as f:
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
        for i in range(len(output_points)):
            f.write("%f %f %f %d %d %d\n" % (output_points[i][0], output_points[i][1],
                    output_points[i][2], output_colors[i][0], output_colors[i][1], output_colors[i][2]))

def depth_map(imgL, imgR):
    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
    # SGBM Parameters -----------------
    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    window_size = 3

    left_matcher = cv.StereoSGBM_create(
        minDisparity=-1,  # normally expect 0
        numDisparities=5*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,  # 3-11 and odd
        P1=8 * 3 * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size,  # P2>P1
        disp12MaxDiff=12,
        uniquenessRatio=10,  # 5-15
        speckleWindowSize=50,  # 50-200
        speckleRange=32,  # 16or32
        preFilterCap=63,
        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 7000
    sigma = 1.5
    alpha = 100
    visual_multiplier = 6

    wls_filter = cv.ximgproc.createDisparityWLSFilter(
        matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR) 
    dispr = right_matcher.compute(imgR, imgL) 
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    # filteredImg = wls_filter.filter(displ, imgL, None, dispr)
    filteredImg = wls_filter.filter(displ, imgL, alpha, dispr)

    filteredImg = cv.normalize(
        src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)

    return filteredImg


def calibrate_camera(images_folder):
    images_names = sorted(glob.glob(images_folder))
    images = []
    for imname in images_names:
        im = cv.imread(imname, 1)
        images.append(im)

    # criteria used by checkerboard pattern detector.
    # Change this if the code can't find the checkerboard
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    rows = 6  # number of checkerboard rows.
    columns = 8  # number of checkerboard columns.
    world_scaling = 1.  # change this to the real world square size. Or not.

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]

    # Pixel coordinates of checkerboards
    imgpoints = []  # 2d points in image plane.

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3d point in real world space

    for frame in images:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # find the checkerboard
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)
        print(ret)
        if ret == True:

            # Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            # opencv can attempt to improve the checkerboard coordinates
            corners = cv.cornerSubPix(
                gray, corners, conv_size, (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows, columns), corners, ret)
            cv.imshow('img', frame)
            k = cv.waitKey(500)

            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, (width, height), None, None)
    print('rmse:', ret)

    return mtx, dist


def stereo_calibrate(mtx1, dist1, mtx2, dist2, frames_folder):
    # read the synched frames
    images_names = glob.glob(frames_folder)
    # images_names = sorted(images_names)
    # sort by number, for example 1.bmp 2.bmp 3.bmp ... 10.bmp instead of 1.bmp 10.bmp 2.bmp ... converting from string to int.
    images_names = sorted(images_names, key=lambda x: int(
        x.split('/')[-1].split('.')[0]))
    print("images_names", images_names)
    c1_images_names = images_names[:len(images_names)//2]
    c2_images_names = images_names[len(images_names)//2:]

    c1_images = []
    print("c1_images_names", c1_images_names)
    print("c2_images_names", c2_images_names)
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv.imread(im1, 1)
        c1_images.append(_im)

        _im = cv.imread(im2, 1)
        c2_images.append(_im)

    # change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    rows = 6  # number of checkerboard rows.
    columns = 8  # number of checkerboard columns.

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)

    # frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]

    # Pixel coordinates of checkerboards
    imgpoints_left = []  # 2d points in image plane.
    imgpoints_right = []

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3d point in real world space

    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(
            gray1, (rows, columns), None)
        c_ret2, corners2 = cv.findChessboardCorners(
            gray2, (rows, columns), None)

        if c_ret1 == True and c_ret2 == True:
            corners1 = cv.cornerSubPix(
                gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(
                gray2, corners2, (11, 11), (-1, -1), criteria)

            cv.drawChessboardCorners(frame1, (rows, columns), corners1, c_ret1)
            cv.imshow('img', frame1)

            cv.drawChessboardCorners(frame2, (rows, columns), corners2, c_ret2)
            cv.imshow('img2', frame2)
            k = cv.waitKey(500)

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    # stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    stereocalibration_flags = cv.CALIB_USE_INTRINSIC_GUESS
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
                                                                 mtx2, dist2, (width, height), criteria=criteria, flags=stereocalibration_flags)

    print("ret", ret)
    return ret, CM1, dist1, CM2, dist2, R, T, E, F, width, height


# mtx1, dist1 = calibrate_camera(images_folder='left_2/*')
# mtx2, dist2 = calibrate_camera(images_folder='right_2/*')
# save_single_camera_params(mtx1, dist1, 'left_camera_params.yaml')
# save_single_camera_params(mtx2, dist2, 'right_camera_params.yaml')
#read from yaml
fs = cv.FileStorage("left_camera_params.yaml", cv.FILE_STORAGE_READ)
mtx1 = fs.getNode("K").mat()
dist1 = fs.getNode("D").mat()
fs = cv.FileStorage("right_camera_params.yaml", cv.FILE_STORAGE_READ)
mtx2 = fs.getNode("K").mat()
dist2 = fs.getNode("D").mat()

# ret, K1, D1, K2, D2, R, T, E, F, width, height = stereo_calibrate(
#     mtx1, dist1, mtx2, dist2, 'synched/*')
# save_stereo_params(K1, D1, K2, D2, R, T, E, F,width,height, 'stereo_params.yaml')
#read from yaml
fs = cv.FileStorage("stereo_params.yaml", cv.FILE_STORAGE_READ)
K1 = fs.getNode("K1").mat()
D1 = fs.getNode("D1").mat()
K2 = fs.getNode("K2").mat()
D2 = fs.getNode("D2").mat()
R = fs.getNode("R").mat()
T = fs.getNode("T").mat()
E = fs.getNode("E").mat()
F = fs.getNode("F").mat()
width = int(fs.getNode("width").real())
height = int(fs.getNode("height").real())


#! change alpha here
# Q is the disparity-to-depth mapping matrix, which is a 4 x 4 valid disparity-to-depth mapping matrix
R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(
    K1, D1, K2, D2, (width, height), R, T, flags=cv.CALIB_ZERO_DISPARITY, alpha=1.1)
save_rectification_params(R1, R2, P1, P2, Q, roi1, roi2, 'rectification_params.yaml')
#read from yaml
fs = cv.FileStorage("rectification_params.yaml", cv.FILE_STORAGE_READ)
R1 = fs.getNode("R1").mat()
R2 = fs.getNode("R2").mat()
P1 = fs.getNode("P1").mat()
P2 = fs.getNode("P2").mat()
Q = fs.getNode("Q").mat()
roi1 = fs.getNode("roi1").mat()
roi2 = fs.getNode("roi2").mat()

# chessboard 
# leftFrame = cv.imread("left_2/l1.bmp")
# rightFrame = cv.imread("right_2/r1.bmp")
# pot 
leftFrame = cv.imread("l10.bmp")
rightFrame = cv.imread("r10.bmp")
# perform radiometric calibration
matched = radiometric_calibration(leftFrame, rightFrame)
#plot 
#title 
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3,
                                    figsize=(8, 3),
                                    sharex=True, sharey=True)
plt.tight_layout()
for aa in (ax1, ax2, ax3):
    aa.set_axis_off()
ax1.imshow(rightFrame)
ax1.set_title('right')
ax2.imshow(leftFrame)
ax2.set_title('left')
ax3.imshow(matched)
ax3.set_title('radio')
#save radio
cv.imwrite("radio.bmp",matched)

# histogram matching to resolve color differences

image = matched
reference = leftFrame
  
matched = match_histograms(image, reference , 
                          channel_axis=-1) 

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3,
                                    figsize=(8, 3),
                                    sharex=True, sharey=True)
plt.tight_layout()
for aa in (ax1, ax2, ax3):
    aa.set_axis_off()

ax1.imshow(image)
ax1.set_title('Source')
ax2.imshow(reference)
ax2.set_title('Reference')
ax3.imshow(matched)
ax3.set_title('Matched')

plt.show()
#save result of matched
cv.imwrite("matched.bmp",matched)

# undistort
# leftMapX, leftMapY = cv.initUndistortRectifyMap(
#     K1, D1, R1, P1, (width, height), cv.CV_32FC1)
# left_rectified = cv.remap(leftFrame, leftMapX, leftMapY,
#                           cv.INTER_LINEAR, cv.BORDER_CONSTANT)
# rightMapX, rightMapY = cv.initUndistortRectifyMap(
#     K2, D2, R2, P2, (width, height), cv.CV_32FC1)
# right_rectified = cv.remap(rightFrame, rightMapX,
#                            rightMapY, cv.INTER_LINEAR, cv.BORDER_CONSTANT)
# # we need grayscaled for disparity map
# left_gray = cv.cvtColor(left_rectified, cv.COLOR_BGR2GRAY)
# right_gray = cv.cvtColor(right_rectified, cv.COLOR_BGR2GRAY)
# disparity_image = depth_map(left_gray, right_gray)
# # generate point cloud
# output_file = "point_cloud.ply"
# generate_point_cloud(left_rectified, right_rectified, Q, K1, D1, K2, D2, R1, R2, P1, P2, output_file)

