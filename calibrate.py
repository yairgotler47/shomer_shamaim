import cv2
import numpy as np


def calibrate_camera(images):

    # criteria used by checkerboard pattern detector.
    # Change this if the code can't find the checkerboard
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    rows = 10  # number of checkerboard rows.
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
    used_indexes = []
    for i in range(len(images)):
        cv2.imshow('img', images[i])
        cv2.waitKey(0)
        gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)

        # find the checkerboard
        # ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)
        ret, corners = cv2.findChessboardCornersSB(
            gray, (rows, columns), flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        )
        if ret == True:
            # Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)
            used_indexes.append(i)
            # opencv can attempt to improve the checkerboard coordinates
            # corners = cv2.cornerSubPix(
            #     gray, corners, conv_size, (-1, -1), criteria)
            cv2.drawChessboardCorners(images[i], (rows, columns), corners, ret)
            cv2.imshow('img', images[i])
            cv2.waitKey(500)

            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (width, height), None, None)
    print('rmse:', ret)
    print('camera matrix:\n', mtx)
    print('distortion coeffs:', dist)
    print('Rs:\n', rvecs)
    print('Ts:\n', tvecs)

    return mtx, dist, used_indexes


def stereo_calibrate(mtx1, dist1, mtx2, dist2, left_cal_images, right_cal_images):
    # change this if stereo calibration not good.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    rows = 10  # number of checkerboard rows.
    columns = 7  # number of checkerboard columns.
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
    counter = 0
    for frame1, frame2 in zip(left_cal_images, right_cal_images):
        cv2.imshow('img', frame1)
        cv2.imshow('img2', frame2)
        cv2.waitKey(0)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv2.findChessboardCornersSB(
            gray1, (rows, columns), flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        )

        c_ret2, corners2 = cv2.findChessboardCornersSB(
            gray2, (rows, columns), flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        )
        # c_ret1, corners1 = cv2.findChessboardCorners(
        #     gray1, (rows, columns), None)
        # c_ret2, corners2 = cv2.findChessboardCorners(
        #     gray2, (rows, columns), None)
        print(c_ret1, c_ret2)
        if c_ret1 == True and c_ret2 == True:
            # corners1 = cv2.cornerSubPix(
            #     gray1, corners1, (11, 11), (-1, -1), criteria)
            # corners2 = cv2.cornerSubPix(
            #     gray2, corners2, (11, 11), (-1, -1), criteria)

            cv2.drawChessboardCorners(
                frame1, (rows, columns), corners1, c_ret1)
            cv2.imshow('img', frame1)

            cv2.drawChessboardCorners(
                frame2, (rows, columns), corners2, c_ret2)
            cv2.imshow('img2', frame2)
            cv2.waitKey(500)

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
            counter += 1
    print(counter)
    stereo_calibration_flags = cv2.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1,
                                                                  dist1,
                                                                  mtx2, dist2, (
                                                                      width, height), criteria=criteria,
                                                                  flags=stereo_calibration_flags)

    print(ret)
    return R, T


def write_mat(mat, filename):
    with open(filename, "w") as file:
        for row in mat:
            for i in range(len(row)):
                if i == len(row)-1:
                    file.write(str(row[i])+"\n")
                else:
                    file.write(str(row[i]) + " ")
    return


def read_mat(filename):
    mat = []
    with open(filename, "r") as file:
        lines = file.readlines()
        for line in lines:
            vals = [float(v) for v in line.strip().split()]
            mat.append(vals)
    return np.array(mat, dtype=np.float64)


def write_params(mtx1, mtx2, dist1, dist2, r, t):
    write_mat(mtx1, "mtx1.txt")
    write_mat(dist1, "dist1.txt")
    write_mat(mtx2, "mtx2.txt")
    write_mat(dist2, "dist2.txt")
    write_mat(r, "R.txt")
    write_mat(t, "T.txt")


def get_params():
    mtx1 = read_mat("mtx1.txt")
    dist1 = read_mat("dist1.txt")
    mtx2 = read_mat("mtx2.txt")
    dist2 = read_mat("dist2.txt")
    r = read_mat("R.txt")
    t = read_mat("T.txt")
    return mtx1, mtx2, dist1, dist2, r, t
