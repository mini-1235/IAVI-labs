import cv2
import numpy as np
import os
import glob
import yaml

# 定义棋盘格大小
CHECKERBOARD = (6, 8)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 创建储存3D点的数组和储存2D点的数组
objpoints = []
imgpoints = []

# 定义世界坐标系中的3D点
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                          0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# 读取所有图片
# 列表中第一张图片为144402
images = glob.glob('./lab3/*.jpg')
cnt = 0
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 寻找棋盘格角点
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    """
    如果检测到所需数量的角，就细化像素坐标并显示在棋盘的图像上
    """
    if ret == True:
        cnt += 1
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
        
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        # cv2.imwrite('./cornered/result' + str(cnt) + '.jpg', img)

h,w = img.shape[:2]
 
"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
 
print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)

# 将标定结果保存到result.yml文件中
data = {'camera_matrix': np.asarray(mtx).tolist(),
        'dist_coeff': np.asarray(dist).tolist()}
with open("./result.yml", "w") as f:
    yaml.dump(data, f)


img = cv2.imread('./lab3/IMG_20231008_144402.jpg')
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# 去畸变
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# 根据前面ROI区域裁剪图片
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('./undistorted.jpg', dst)

# 将棋盘与相机中心输出在.ply文件中
square_size = 21.2  # 棋盘格方块大小，单位为毫米

# 计算不同照片中的相机中心
C = []
for i in range(cnt):
    R, _ = cv2.Rodrigues(rvecs[i])
    t = tvecs[i]
    C.append(-np.dot(R.T, t))

# 计算棋盘格角点的世界坐标
corners_3d = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
corners_3d[:, :2] = objp[0, :, :2]
corners_3d[:, 2] = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1]))
corners_3d = corners_3d

# 输出相机中心和棋盘格角点到 .ply 文件
with open('output.ply', 'w') as f:
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write('element vertex %d\n' % (CHECKERBOARD[0] * CHECKERBOARD[1] + cnt))
    f.write('property float x\n')
    f.write('property float y\n')
    f.write('property float z\n')
    f.write('property uchar red\n')
    f.write('property uchar green\n')
    f.write('property uchar blue\n')
    f.write('element face %d\n' % (CHECKERBOARD[0]/2 * CHECKERBOARD[1]/2 + (CHECKERBOARD[0]-2)/2 * (CHECKERBOARD[1]-2)/2))
    f.write('property list uchar int vertex_indices\n')
    f.write('end_header\n')
    # 输出棋盘格角点
    for i in range(CHECKERBOARD[0] * CHECKERBOARD[1]):
        f.write('%f %f %f 0 0 0\n' % (corners_3d[i, 0], corners_3d[i, 1], corners_3d[i, 2]))
    # 输出相机中心
    for i in range(cnt):
        f.write('%f %f %f 0 0 0\n' % tuple(C[i]))
    # 输出棋盘格面（正方形）
    # 只输出黑色的面
    for i in range(CHECKERBOARD[0]):
        for j in range(CHECKERBOARD[1]):
            if ((i+j) % 2 == 0) and (i != 5) and (j != 7):
                # print('4 %d %d %d %d\n' % (i + j*CHECKERBOARD[0], i+1 + j*CHECKERBOARD[0], i+1 + (j+1)*CHECKERBOARD[0], i + (j+1)*CHECKERBOARD[0]))
                f.write('4 %d %d %d %d\n' % (i + j*CHECKERBOARD[0], i+1 + j*CHECKERBOARD[0], i+1 + (j+1)*CHECKERBOARD[0], i + (j+1)*CHECKERBOARD[0]))
print('输出完成')

# 读取需要投射的模型文件
with open('./model.ply', 'r') as f:
    lines = f.readlines()

# 获取点的数量和起始行号
for i, line in enumerate(lines):
    if line.startswith('element vertex'):
        num_points = int(line.split()[2])
        start_line = i + 5
        break

# 读取点的坐标
points = []
for i in range(start_line, start_line + num_points):
    x, y, z = map(float, lines[i].split()[:3])
    points.append([x, y, z])

print('读取完成')

# 将点转换到图像坐标系中
points = np.array(points)
imgpoints2, _ = cv2.projectPoints(points, rvecs[0], tvecs[0], mtx, dist)

# 将点叠加到原图片中
img = cv2.imread('./lab3/IMG_20231008_144402.jpg')
for i in range(len(imgpoints2)):
    x, y = imgpoints2[i][0]
    x, y = int(x), int(y)
    cv2.circle(img, (x, y), 1, (0, 0, 255), -1)


# 保存图片
cv2.imwrite('./projection.jpg', img)