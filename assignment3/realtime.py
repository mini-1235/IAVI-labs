import numpy as np
import cv2
import yaml

Grid_Point_Row=6 
Grid_Point_Col=8
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001) # termination criteria
corner_accuracy = (11,11)
result_file = "./calibration.yaml"
object_points = [] # 3d point in real world space
image_points = [] # 2d points in image plane.
tot_error=0
chessboard_points = np.zeros((Grid_Point_Row*Grid_Point_Col,3), np.float32)
chessboard_points[:,:2] = np.mgrid[0:Grid_Point_Row,0:Grid_Point_Col].T.reshape(-1,2)

# Intialize camera and window
camera = cv2.VideoCapture(0) #Supposed to be the only camera
if not camera.isOpened():
    print("Camera not found!")
    quit()
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))  
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
cv2.namedWindow("Calibration")

# Read the model
with open('./model.ply', 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if line.startswith('element vertex'):
        num_points = int(line.split()[2])
        start_line = i + 5
        break

points = []
for i in range(start_line, start_line + num_points):
    x, y, z = map(float, lines[i].split()[:3])
    points.append([x, y, z])
points = np.array(points)

while True:    
    # Read from camera and display on windows
    ret, img = camera.read()
    cv2.imshow("Calibration", img)
    
    # Wait for instruction 
    k = cv2.waitKey(50) 
   
    # SPACE pressed to take picture
    if k%256 == 32:   
        print("Adding image for calibration...")
        imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(imgGray, (Grid_Point_Row,Grid_Point_Col),None)

        # If found, add object points, image points (after refining them)
        if not ret:
            print("Cannot found Chessboard corners!")
            
        else:
            print("Chessboard corners successfully found.")
            object_points.append(chessboard_points)
            corners2 = cv2.cornerSubPix(imgGray,corners,(11,11),(-1,-1),criteria)
            image_points.append(corners2)
            # Draw and display the corners
            imgAugmnt = cv2.drawChessboardCorners(img, (Grid_Point_Row,Grid_Point_Col), corners2,ret)
            cv2.imshow('Calibration',imgAugmnt) 
            cv2.waitKey(500)        

    # C pressed to compute calibration             
    elif k%256 == 99: 
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, (width,height),None,None)
        
        if not ret:
            print("Cannot compute calibration!")
        
        else:
            print("Camera calibration successfully computed")
            # Compute reprojection errors
            for i in range(len(object_points)):
                image_points2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(image_points[i],image_points2, cv2.NORM_L2)/len(image_points2)
                tot_error += error
            print("Camera matrix: ", mtx)
            print("Distortion coeffs: ", dist)
            print("Total error: ", tot_error)
            print("Mean error: ", np.mean(error))
            
            # Saving calibration matrix
            print("Saving camera matrix .. in ",result_file)
            data={"camera_matrix": mtx.tolist(), "dist_coeff": dist.tolist()}
            with open(result_file, "w") as f:
                yaml.dump(data, f, default_flow_style=False)

    # S pressed to project the model on image
    elif k%256 == 115:
        imgpoints, _ = cv2.projectPoints(points, rvecs[i], tvecs[i], mtx, dist)
        for i in range(len(imgpoints)):
            x, y = imgpoints[i][0]
            x, y = int(x), int(y)
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        print("Projecting model on image...")
        cv2.imshow('Calibration', img)
        cv2.waitKey(500)