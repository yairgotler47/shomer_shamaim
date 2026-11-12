import cv2
import tools
import numpy as np
CAL_IMAGE_NUM=15

def capture_and_calibrate_stereo(cap_left,cap_right):
    left_images,right_images=tools.take_n_dual_pictures(cap_left,cap_right,CAL_IMAGE_NUM)
    mtx1,dist1=tools.calibrate_camera(left_images)
    mtx2,dist2=tools.calibrate_camera(right_images)

    r,t=tools.stereo_calibrate(mtx1,dist1,mtx2,dist2,left_images,right_images)
    tools.write_params(mtx1,mtx2,dist1,dist2,r,t)
def process_video(camera):
    tools.take_a_picture(camera,"background")
    background = cv2.imread("background")
    cv2.imshow("background",background)
    cv2.waitKey(0)
    cv2.destroyWindow("background")
    count, success = 0, True
    center_points=[]
    while success:
        success, image = camera.read() # Read frame
        if success and count%5==0:
            #subtract frame
            image=cv2.subtract(image,background)
            #sharpen frame
            image=tools.sharpen_image(image)
            #find ball center
            x,y=tools.find_object_center(image)
            center_points.append([x,y])
        count += 1
        if count==100:
            break
    return center_points

def process_stereo_video(left_cap,right_cap):
    mtx1, mtx2, dist1, dist2, r, t = tools.get_params()
    left_ret,left_back=left_cap.read()
    right_ret,right_back=right_cap.read()
    cv2.imshow("left background",left_back)
    cv2.imshow("right background",right_back)
    cv2.waitKey(0)
    cv2.destroyWindow("left background")
    cv2.destroyWindow("right background")
    count,success = 0,True
    center_points=[]

    while left_ret and right_ret:
        left_ret, left_frame = left_cap.read()
        right_ret, right_frame = right_cap.read()
        if count%5==0:# Read frame
            left_point=[]
            right_point=[]
            if left_ret:
                #subtract frame
                # left_frame=cv2.subtract(left_frame,left_back)
                #sharpen frame
                left_frame=tools.sharpen_image(left_frame)
                #find ball center
                left_point=tools.find_object_center(left_frame)

            if right_ret:
                #subtract frame
                # right_frame=cv2.subtract(right_frame,right_back)
                #sharpen frame
                right_frame=tools.sharpen_image(right_frame)
                #find ball center
                right_point=tools.find_object_center(right_frame)
            if left_point and right_point:
                center_points.append((tools.triangulate_single(mtx1,mtx2,left_point,right_point,r,t),count))
        count += 1
        if count==100:
            break

    return center_points

left_cap=cv2.VideoCapture(1)
right_cap=cv2.VideoCapture(2)
print(process_stereo_video(left_cap, right_cap))
