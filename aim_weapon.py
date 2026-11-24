import cv2
import numpy as np
import serial
import time
import math

import capture
import triangulate

SPD = 1000
TIKES = 800
DIR = 1
ser = serial.Serial('COM13', 115200, timeout=0.01)
MIN_RAD_ROT = (np.pi*0)/100
MAX_RAD_ROT = (np.pi*45)/100

def send_angle_command(dir1, theta, dir2=0, phi=0, power_rifle=0):
    print("hi")
    cmd = str(dir1) + ';' + str(theta).zfill(3) + ';' + str(dir2) + ";" + str(phi).zfill(3) + ";" + str(power_rifle)
    print(cmd)
    # time.sleep(2)
    ser.write(cmd.encode())

def calc_rotation(point3d, translation_vec):
    curr_point = [point3d[0] + translation_vec[0], point3d[1] + translation_vec[1], point3d[2] + translation_vec[2]]
    lat = np.arctan(curr_point[1] / curr_point[2])
    pitch = np.arctan(curr_point[0] / curr_point[2])
    return lat, pitch

def send_point_command(translation_vec, point3d):

    theta, phi = calc_rotation(point3d, translation_vec)
    if (phi< MIN_RAD_ROT or phi > MAX_RAD_ROT):
        print("point out of range")
        return

    power_rifle = 0
    dir1 = 0
    if (theta < 0):
        dir1 = 1
    dir2 = 1
    # if(phi<0):
    #     dir2 = 1

    send_angle_command(dir1, abs(theta), dir2, abs(phi), power_rifle)
    return

def point_capture(event, x, y, flags, param):
    state = param
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Coordinate of Image: ({x}, {y})")
        state["point"] = (x, y)
        cv2.circle(state["image"], (x, y), radius=2, color=(180,180,255), thickness=-1)
        cv2.putText(state["image"], f'({x}, {y})', (x+8, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,255), 2)
        cv2.imshow('image', state["image"])

def pick_point_with_mouse(img):
    state = {
        "image": img.copy(),
        "point": None,
    }
    cv2.imshow("image", state["image"])
    cv2.setMouseCallback('image', point_capture, param = state)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(state["point"])
    return state["point"]

def aim_at_clicked_point(cam_l,cam_r, mtx1, mtx2, dist1, dist2, pt1, pt2, R, T, translation_vec):
    l_frame, _, r_frame ,_ = capture.read_stereo_frames(cam_l,cam_r)
    if l_frame is None or r_frame is None:
        return False

    pt1 = pick_point_with_mouse(l_frame)
    pt2 = pick_point_with_mouse(r_frame)
    if pt1 is None or pt2 is None:
        return True
    point3d = triangulate.triangulate_single(mtx1, mtx2, dist1, dist2, pt1, pt2, R, T)
    send_point_command(translation_vec, point3d)
    return True


def main():
    _, frame = capture.take_a_picture(camera_index=0)
    pick_point_with_mouse(frame)

if __name__ == "__main__":
    main()
