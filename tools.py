import cv2
import numpy as np
import matplotlib.pyplot as plt
from cvzone.ColorModule import ColorFinder

left_cal_folder="left_cal_images/"
# capturing
def take_a_picture(cam):
    ret,frame=cam.read()
    if ret:
        cv2.imshow("Image", frame)
        cv2.waitKey(0)
        cv2.destroyWindow("Image")
    else:
        print("failed image capture")

    return ret,frame


def take_dual_picture(cap_left,cap_right):
    # Capture one frame
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if ret_left:
        cv2.imshow("Left Captured", frame_left)
        print("image captured")
        cv2.waitKey(0)
        cv2.destroyWindow("Left Captured")
    else:
        print("Failed to capture left image.")

    if ret_right:
        cv2.imshow("Right Captured", frame_right)
        print("image captured")
        cv2.waitKey(0)
        cv2.destroyWindow("Right Captured")
    else:
        print("Failed to capture right image.")

    return ret_left,frame_left,ret_right,frame_right

def take_n_pictures(camera,n,folder):
    images=[]
    for i in range(n):
        ret,frame=take_a_picture(camera)
        if ret:
            images.append(frame)
    return images

def take_n_dual_pictures(cap_left,cap_right,n):
    left_images=[]
    right_images=[]
    for i in range(n):
        ret_left,frame_left,ret_right,frame_right=take_dual_picture(cap_left,cap_right)
        if ret_left:
            left_images.append(frame_left)
        if ret_right:
            right_images.append(frame_right)
    return left_images,right_images

# camera calibration
def calibrate_camera(images):
    # plt.figure(figsize = (10,10))
    # ax = [plt.subplot(2,2,i+1) for i in range(4)]
    #
    # for a, frame in zip(ax, images):
    #     a.imshow(frame[:,:,[2,1,0]])
    #     a.set_xticklabels([])
    #     a.set_yticklabels([])
    # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.show()

    # criteria used by checkerboard pattern detector.
    # Change this if the code can't find the checkerboard
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    rows = 7  # number of checkerboard rows.
    columns = 7  # number of checkerboard columns.
    world_scaling = 1.8  # change this to the real world square size. Or not.

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # find the checkerboard
        ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)

        if ret == True:
            # Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            # opencv can attempt to improve the checkerboard coordinates
            corners = cv2.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv2.drawChessboardCorners(frame, (rows, columns), corners, ret)
            cv2.imshow('img', frame)
            cv2.waitKey(500)

            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print('rmse:', ret)
    print('camera matrix:\n', mtx)
    print('distortion coeffs:', dist)
    print('Rs:\n', rvecs)
    print('Ts:\n', tvecs)

    return mtx, dist

def stereo_calibrate(mtx1, dist1, mtx2, dist2, left_cal_images, right_cal_images):
    # change this if stereo calibration not good.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    rows = 7 # number of checkerboard rows.
    columns = 7 # number of checkerboard columns.
    world_scaling = 1.8  # change this to the real world square size. Or not.

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # frame dimensions. Frames should be the same size.
    width = left_cal_images[0].shape[1]
    height = left_cal_images[0].shape[0]

    # Pixel coordinates of checkerboards
    imgpoints_left = []  # 2d points in image plane.
    imgpoints_right = []

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3d point in real world space

    for frame1, frame2 in zip(left_cal_images, right_cal_images):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv2.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv2.findChessboardCorners(gray2, (rows, columns), None)

        if c_ret1 == True and c_ret2 == True:
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            cv2.drawChessboardCorners(frame1, (rows, columns), corners1, c_ret1)
            cv2.imshow('img', frame1)

            cv2.drawChessboardCorners(frame2, (rows, columns), corners2, c_ret2)
            cv2.imshow('img2', frame2)
            cv2.waitKey(500)

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    stereo_calibration_flags = cv2.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1,
                                                                 dist1,
                                                                 mtx2, dist2, (width, height), criteria=criteria,
                                                                 flags=stereo_calibration_flags)

    print(ret)
    return R, T
def write_mat(mat,filename):
    with open(filename,"w") as file:
        for row in mat:
            for i in range(len(row)):
                if i==len(row)-1:
                    file.write(str(row[i])+"\n")
                else:
                    file.write(str(row[i]) + " ")
    return

def read_mat(filename):
    mat=[]
    with open(filename,"r") as file:
        lines=file.readlines()
        for line in lines:
            vals=line.split(" ")
            for i in range(len(vals)):
                print(vals[i])
                vals[i]=float(vals[i])
            mat.append(vals)
    return mat

def write_params(mtx1,mtx2,dist1,dist2,r,t):
    write_mat(mtx1,"mtx1.txt")
    write_mat(dist1,"dist1.txt")
    write_mat(mtx2,"mtx2.txt")
    write_mat(dist2,"dist2.txt")
    write_mat(r,"R.txt")
    write_mat(t,"T.txt")

def get_params():
    mtx1=read_mat("mtx1.txt")
    dist1=read_mat("dist1.txt")
    mtx2=read_mat("mtx2.txt")
    dist2=read_mat("dist2.txt")
    r=read_mat("R.txt")
    t=read_mat("T.txt")
    return mtx1,mtx2,dist1,dist2,r,t
# image processing
def sharpen_image(image):
    rows, cols = image.shape[:2]

    # gaussian kernel for sharpening
    gaussian_blur = cv2.GaussianBlur(image,(7,7),sigmaX=2)

    # sharpening using addWeighted()
    sharp1 = cv2.addWeighted(image,1.5,gaussian_blur,-0.5,0)
    sharp2 = cv2.addWeighted(image,3.5,gaussian_blur,-2.5,0)
    sharp3 = cv2.addWeighted(image,7.5,gaussian_blur,-6.5,0)
    # the addweight() method performs a linear comnbination of matrices
    # which is simple arithmetic operations for example the first function will
    # have the resultant matrix as image * 1.5 + gaussian_blur * (-0.5) + 0

    # showing the images
    # cv.imshow('sharp1',sharp1)
    # cv.imshow('sharp2',sharp2)
    # cv.imshow('sharp3',sharp3)
    # cv.imshow('original',image)
    # cv.waitKey(0)
    return sharp1

# object detection
def find_object_center(img):

    # Webcam or video of a basketball shot

    myColorFinder = ColorFinder(False)
    hsvVals = {'hmin':0, 'smin': 93, 'vmin':110, 'hmax': 35, 'smax':201, 'vmax':255}

    imgColor, mask =myColorFinder.update(img, hsvVals)
    img = cv2.resize(img, (1280, 720))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Orange basketball mask (tune for your lighting)
    lower = np.array([10, 100, 100])
    upper = np.array([100, 200, 200])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)


    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius > 5:
            center = (int(x), int(y))
            cv2.circle(img, center, int(radius), (0, 255, 0), 2)
            print(x,y)
            cv2.imshow('TechVidvan', img)
            cv2.waitKey(0)
            return x,y
    return

# object position

def DLT(P1, P2, point1, point2):

    A = [point1[1] * P1[2, :] - P1[1, :],
             P1[0, :] - point1[0] * P1[2, :],
             point2[1] * P2[2, :] - P2[1, :],
             P2[0, :] - point2[0] * P2[2, :]
             ]
    A = np.array(A).reshape((4, 4))
    # print('A: ')
    # print(A)

    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices=False)

    print('Triangulated point: ')
    print(Vh[3, 0:3] / Vh[3, 3])
    return Vh[3, 0:3] / Vh[3, 3]

def triangulate(mtx1, mtx2,image_points1,image_points2, R, T):
    image_points1 = np.array(image_points1)
    image_points2 = np.array(image_points2)
    # RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
    P1 = mtx1 @ RT1  # projection matrix for C1

    # RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([R, T], axis=-1)
    P2 = mtx2 @ RT2 # projection matrix for C2

    p3ds=[]
    for i in range(len(image_points1)):
        p3ds.append(DLT(P1, P2, image_points1[i], image_points2[i]))
    return p3ds


def triangulate_single(mtx1, mtx2,image_point1,image_point2,R, T):
    # RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
    P1 = mtx1 @ RT1  # projection matrix for C1

    # RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([R, T], axis=-1)
    P2 = mtx2 @ RT2  # projection matrix for C2


    p3d = DLT(P1, P2, image_point1, image_point2)

    return p3d


