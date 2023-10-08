

import numpy as np
import cv2
import glob
import json
import sys 

Grid_Point_Col = 8
Grid_Point_Row = 6
# Set termination criteria. We stop either when an accuracy is reached or when
# we have finished a certain number of iterations.
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 40, 0.001)

chessboard_points = np.zeros((Grid_Point_Col * Grid_Point_Row, 3), np.float32)
chessboard_points[:, :2] = np.mgrid[0:Grid_Point_Col, 0:Grid_Point_Row].T.reshape(-1, 2)

image_points = []
object_points = []
loop_count = 0
image_paths = glob.glob('dataset/lab3/*.jpg')
image_size = None
# print(sys.argv[1])
for image_path in image_paths:
    print(loop_count)
    # if(loop_count == sys.argv[1]):
    #     break
    print(image_path)
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_size = image_gray

    is_chessboard_found, corners = cv2.findChessboardCorners(image_gray, (Grid_Point_Col, Grid_Point_Row))

    if is_chessboard_found:
        cv2.cornerSubPix(image_gray, corners, (11, 11), (-1, -1), criteria)
        image_points.append(corners)
        object_points.append(chessboard_points)
        cv2.drawChessboardCorners(image, (Grid_Point_Col, Grid_Point_Row), corners, is_chessboard_found)
        cv2.namedWindow("Chessboard Calibration", 0)
        cv2.resizeWindow("Chessboard Calibration", 1280, 1024)
        cv2.imshow('Chessboard Calibration', image)
        cv2.imwrite('processing.jpg', image)
        cv2.waitKey(200)
        loop_count += 1

    else:
        print(image_path)
        pass
        print("Chessboard Not Detectable!")

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size.shape[::-1], None, None)


print("\n------RMS:------\n")
print(ret)
print("\n------Camera Matrix:------\n")
json_mtx = json.dumps(mtx.tolist())
print(json_mtx)
print("\n\n------Distortion Coefficient:------\n")
json_dist = json.dumps(dist.tolist())
print(json_dist)
print("\n\n------Rotation Matrix:------\n")
for rvec in rvecs:
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    print(rotation_matrix)
print("\n\n------Translation Vector:------\n")
for tvec in tvecs:
    tranlsation_matrix,_ =  cv2.Rodrigues(tvec)
    print(tranlsation_matrix)

image_before_undistort = cv2.imread('dataset/lab3/IMG_20231008_163510.jpg')
h, w = image_before_undistort.shape[:2]
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
image_after_undistort = cv2.undistort(image_before_undistort, mtx, dist, None, new_camera_matrix)

x, y, w, h = roi
image_after_undistort = image_after_undistort[y:y + h, x:x + w]
cv2.imwrite('result.jpg', image_after_undistort)

total_error = 0
for i in range(len(object_points)):
    img_points2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(image_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
    total_error += error

mean_error = total_error / len(object_points)
print("\n\n------total error:------\n\n", total_error)
print("\n\n------mean error:------\n\n", mean_error)
# print(mean_error)