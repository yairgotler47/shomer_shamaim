from copy import deepcopy
import time
import cv2
import numpy as np
import calibrate
import capture
import trajectory
import triangulate
import image_process

CAL_IMAGE_NUM = 15
THROW_SPEED_THRESHOLD = 150.0  # cm/s
MIN_FLIGHT_FRAMES = 4


def capture_and_calibrate_stereo(cap_left, cap_right):
    left_images, right_images = capture.take_n_dual_pictures(
        cap_left, cap_right, CAL_IMAGE_NUM
    )
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
    r, t = calibrate.stereo_calibrate(
        mtx1, dist1, mtx2, dist2, left_to_use, right_to_use
    )
    calibrate.write_params(mtx1, mtx2, dist1, dist2, r, t)


def process_stereo_video(left_cap, right_cap):
    state = np.zeros(6, dtype=np.float64)  # x, y, z, vx, vy, vz
    H, Q, R, P = trajectory.init_filter_matrices()
    mtx1, mtx2, dist1, dist2, r, t = calibrate.get_params()

    left_ret, left_back, right_ret, right_back = capture.read_stereo_frames(
        left_cap, right_cap
    )
    if not (left_ret and right_ret):
        print("Failed to read initial stereo frames.")
        return

    cv2.imshow("left background", left_back)
    cv2.imshow("right background", right_back)
    cv2.waitKey(0)
    cv2.destroyWindow("left background")
    cv2.destroyWindow("right background")

    count = 0
    flight_frames = 0
    in_flight = False
    prev_point = None
    prev_time = None
    last_step_time = None
    center_points = []
    future_points = []

    while True:
        left_ret, left_frame, right_ret, right_frame = capture.read_stereo_frames(
            left_cap, right_cap
        )
        now = time.perf_counter()

        if not (left_ret and right_ret):
            break

        left_point = image_process.find_object_center(left_frame, "left_image", False)
        right_point = image_process.find_object_center(right_frame, "right_image", False)

        if in_flight:
            dt = max(now - last_step_time, 1e-4)
            last_step_time = now
            F, B_u = trajectory.build_motion_matrices(dt)
            state, P = trajectory.predict(state, P, B_u, F, Q)

        if left_point and right_point:
            center_point = triangulate.triangulate_single(
                mtx1, mtx2, dist1, dist2, left_point, right_point, r, t
            )

            if prev_point is None:
                prev_point = center_point
                prev_time = now
            else:
                dt_meas = max(now - prev_time, 1e-4)
                raw_vel = (center_point - prev_point) / dt_meas
                raw_speed = np.linalg.norm(raw_vel)
                prev_point = center_point
                prev_time = now

                if not in_flight:
                    if raw_speed >= THROW_SPEED_THRESHOLD:
                        state = np.array([
                            center_point[0], center_point[1], center_point[2],
                            raw_vel[0], raw_vel[1], raw_vel[2],
                        ], dtype=np.float64)
                        last_step_time = now
                        in_flight = True
                        flight_frames = 1
                        center_points.append(center_point)
                else:
                    state, P = trajectory.update(state, P, H, R, center_point)
                    center_points.append(center_point)
                    flight_frames += 1

                    if flight_frames >= MIN_FLIGHT_FRAMES:
                        future_points = trajectory.predict_trajectory(
                            state, P, Q)

        count += 1
        if count == 3000:
            break

    left_cap.release()
    right_cap.release()
    cv2.destroyAllWindows()
    trajectory.show_3d_graph(center_points, future_points)


left_cap = cv2.VideoCapture(1)
right_cap = cv2.VideoCapture(2)
capture_and_calibrate_stereo(left_cap, right_cap)
process_stereo_video(left_cap, right_cap)
