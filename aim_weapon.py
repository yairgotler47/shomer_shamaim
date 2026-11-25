import cv2
import numpy as np
import serial
import time
import math

import capture
import triangulate
import calibrate

SPD = 1000
TIKES = 800
DIR = 1
ser = serial.Serial('COM13', 115200, timeout=0.01)
MIN_RAD_ROT = (np.pi*0)/100
MAX_RAD_ROT = (np.pi*45)/100
ZERO_TOP = 20

def send_tick_command(dir1, theta, dir2=0, phi=0, power_rifle=0):
    print("hi")
    cmd = str(dir1) + ';' + str(int(round(theta))).zfill(3) + ';' + str(dir2) + ";" + str(int(round(phi))).zfill(3) + ";" + str(power_rifle)
    print(cmd)
    # time.sleep(2)
    ser.write(cmd.encode())

def send_angles(angles_theta_phi: tuple, to_shoot = 0):
    """
    gets angles in rad and send in steps and calc dir
    """
    theta = (angles_theta_phi[0]*50*2)/np.pi
    dir_t = 1 if theta<0 else 0
    phi = (angles_theta_phi[1]*50*2)/np.pi - ZERO_TOP
    dir_p = 1 if phi<0 else 0
    print(f"{dir_t} {theta} {dir_p} {phi}")
    send_tick_command(dir_t, abs(theta), dir_p, abs(phi),to_shoot)


def calc_rotation(point3d, translation_vec):
    curr_point = [point3d[0] + translation_vec[0], point3d[1] + translation_vec[1], point3d[2] + translation_vec[2]]
    lat = np.arctan(curr_point[0] / curr_point[2])
    pitch = np.arctan(curr_point[1] / np.sqrt(curr_point[2]**2+curr_point[0]**2))
    return lat, pitch

def send_point_command(translation_vec, point3d, to_shoot=0):

    theta, phi = calc_rotation(point3d, translation_vec)
    # if (phi< MIN_RAD_ROT or phi > MAX_RAD_ROT):
    #     print("point out of range")
    #     return

    # power_rifle = 0
    # dir1 = 0
    # if (theta < 0):
    #     dir1 = 1
    # dir2 = 1
    # if(phi<0):
    #     dir2 = 1

    send_angles((theta,phi), to_shoot)
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

def aim_at_clicked_point(cam_l, cam_r, mtx1, mtx2, dist1, dist2, R, T, translation_vec, to_shoot):
    _,l_frame,_, r_frame = capture.read_stereo_frames(cam_l,cam_r)
    #l_frame = cv2.imread("lfram.png")
    #r_frame = cv2.imread("rfram.png")
    #print(l_frame,r_frame)
    cv2.imwrite("lfram.png", l_frame)
    cv2.imwrite("rfram.png", r_frame)
    
    if l_frame is None or r_frame is None:
        return False

    pt1 = np.float64(pick_point_with_mouse(l_frame))
    pt2 = np.float64(pick_point_with_mouse(r_frame))
    if pt1 is None or pt2 is None:
        return True
    point3d = triangulate.triangulate_single(mtx1, mtx2, dist1, dist2, pt1, pt2, R, T)

    send_point_command(translation_vec, point3d,to_shoot)
    return True

def pick_point(cam_l, cam_r, mtx1, mtx2, dist1, dist2, R, T):
    _,l_frame,_, r_frame = capture.read_stereo_frames(cam_l,cam_r)
    pt1 = np.float64(pick_point_with_mouse(l_frame))
    pt2 = np.float64(pick_point_with_mouse(r_frame))
    if pt1 is None or pt2 is None:
        return True
    point3d = triangulate.triangulate_single(mtx1, mtx2, dist1, dist2, pt1, pt2, R, T)
    return point3d

def get_trans(cam_l, cam_r, mtx1, mtx2, dist1, dist2, R, T):
    black = pick_point(cam_l, cam_r, mtx1, mtx2, dist1, dist2, R, T)
    red = pick_point(cam_l, cam_r, mtx1, mtx2, dist1, dist2, R, T)
    return [black[0] - red[0], black[1] - red[1], black[2] - red[2] ]


def main():
    # _, frame = capture.take_a_picture(camera_index=0)
    # pick_point_with_mouse(frame)
    #angles = [(180,0),(0,-36), (-100,-10),(50,-15*1.8), (-170,10), (130,0), (1,0), (0,20*1.8)]
    #input("start")
    #for ang in angles:
    #
    #    send_angles((ang[0]* np.pi/180, ang[1]* np.pi/180))
    #   time.sleep(0.5)
    #time.sleep(10)
    # while(True):
    #     theta = int(input("enter theta: ")) * np.pi/180

    #     phi = -1
    #     #while(phi<0 or phi>45):  
    #     phi = int(input("enter phi: ")) * np.pi/180
    #     send_angles((theta, phi))
    #cam_r =  cv2.VideoCapture(1)
    cam_r = None
    cam_l = None

    print(cam_r)
    cam_r =  cv2.VideoCapture(2)
    cam_l =  cv2.VideoCapture(1)
    mtx1, mtx2, dist1, dist2, R, T = calibrate.get_params()
    #pt1 = pick_point_with_mouse(cam_r)
    #pt2 = pick_point_with_mouse()
    #to shoot tv = [-8,20,13]
    translation_vec = [-13,-5.3,13]
    while True:
        send_tick_command(0,0,0,0,0)
        to_shoot = int(input("enter one to make gun shoot: "))
        aim_at_clicked_point(cam_l, cam_r, mtx1, mtx2, dist1, dist2, R, T, translation_vec, to_shoot)

        chang = input("change: ")
        if(chang):
            a = float(input("1: "))
            b= float(input("2: "))
            c = float(input("3: "))
            translation_vec = [a,b,c]
            #translation_vec = get_trans(cam_l, cam_r, mtx1, mtx2, dist1, dist2, R, T)
            
        # if(chang == '1'):
        #     a = float(input("1: "))
        #     b= float(input("2: "))
        #     c = float(input("3: "))
        #     translation_vec = [a,b,c]
            
    aim_at_clicked_point(cam_l, cam_r, mtx1, mtx2, dist1, dist2, R, T, translation_vec)

    

if __name__ == "__main__":
    main()
