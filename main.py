from copy import deepcopy
import cv2
import calibrate
import capture
import trajectory
import triangulate
import image_process

CAL_IMAGE_NUM = 15


def capture_and_calibrate_stereo(cap_left, cap_right):
    left_images, right_images = capture.take_n_dual_pictures(cap_left, cap_right, CAL_IMAGE_NUM)
    copy_left = deepcopy(left_images)
    copy_right = deepcopy(right_images)
    mtx1, dist1, used1 = calibrate.calibrate_camera(copy_left)
    mtx2, dist2, used2 = calibrate.calibrate_camera(copy_right)
    left_to_use = []
    right_to_use = []
    for i in range(len(left_images)):
        if i in used1 and i in used2:
            left_to_use.append(left_images[i])
            right_to_use.append(right_images[i])
    r, t = calibrate.stereo_calibrate(mtx1, dist1, mtx2, dist2, left_to_use, right_to_use)
    calibrate.write_params(mtx1, mtx2, dist1, dist2, r, t)


def process_stereo_video(left_cap, right_cap):
    state = [0, 0, 0, 0, 0, 0]  # x,y,z,vx,vy,vz
    F, B_u, H, Q, R, P = trajectory.define_matrices()
    mtx1, mtx2, dist1, dist2, r, t = calibrate.get_params()
    left_ret, left_back = left_cap.read()
    right_ret, right_back = right_cap.read()
    cv2.imshow("left background", left_back)
    cv2.imshow("right background", right_back)
    cv2.waitKey(0)
    cv2.destroyWindow("left background")
    cv2.destroyWindow("right background")
    count, success = 0, True
    center_points = []
    min_observed_steps = 5
    future_points = []

    while left_ret and right_ret:
        # check time stamps
        left_ret, left_frame = left_cap.read()
        right_ret, right_frame = right_cap.read()
        if count % 1 == 0:  # Read frame
            left_point = []
            right_point = []
            if left_ret:
                # subtract frame
                # left_frame=cv2.subtract(left_frame,left_back)
                # sharpen frame
                # left_frame=tools.sharpen_image(left_frame)
                # find ball center
                left_point = image_process.find_object_center(left_frame, "left_image")

            if right_ret:
                # subtract frame
                # right_frame=cv2.subtract(right_frame,right_back)
                # sharpen frame
                # right_frame=tools.sharpen_image(right_frame)
                # find ball center
                right_point = image_process.find_object_center(right_frame, "right_image")
            if left_point and right_point:
                center_point = triangulate.triangulate_single(mtx1, mtx2, left_point, right_point, r, t)
                center_points.append(center_point)
                if count == 0:
                    state = [center_point[0], center_point[1], center_point[2], 0, 0, 0]
                else:
                    state_pred, P_pred = trajectory.predict(state, P, B_u, F, Q)
                    state, P = trajectory.update(state_pred, P_pred, H, R, center_point)
                if count >= 10 and not future_points:
                    future_points = trajectory.predict_trajectory(state, P, F, B_u, Q)
                    print(f"future points: {future_points}")
        count += 1
        if count == 200:
            break

    trajectory.show_3d_graph(center_points, future_points)


left_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
right_cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
# capture_and_calibrate_stereo(left_cap,right_cap)
print(process_stereo_video(left_cap, right_cap))
