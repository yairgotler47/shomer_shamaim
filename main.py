import cv2
import tools
CAL_IMAGE_NUM=15
def capture_and_calibrate_camera(camera):
    return
def capture_and_calibrate_stereo(cap_left,cap_right):
    left_images,right_images=tools.take_n_dual_pictures(cap_left,cap_right,CAL_IMAGE_NUM)
def process_video(camera):
    ret_back,background = camera.read()
    cv2.imshow("background",background)
    cv2.waitKey(0)
    cv2.destroyWindow("background")
    count, success = 0, True
    center_points=[]
    while success:
        success, image = camera.read() # Read frame
        if success and count%1==0:
            #cv2.imwrite("f" + str(count) + ".png", image)

            #subtract frame
            image=cv2.subtract(image,background)
            #cv2.imwrite("AAAA7"+str(count) + ".png", image)

            #sharpen frame
            image=tools.sharpen_image(image)
            #find ball center
            point=tools.find_object_center(image, "f"+str(count))
            if point:
                center_points.append(point)
        count += 1
        if count==100:
            break
    return center_points

def main():
    print("start")
    cam = cv2.VideoCapture(1)
    cam.set(3, 640)
    cam.set(4, 360)
    print("!!")
    points = process_video(cam)
    print(points)
    #mtx1, dist1 = tools.calibrate_camera(images_folder='left_cal_images/*')
    #mtx2, dist2 = tools.calibrate_camera(images_folder='right_cal_images/*')

    #R, T = tools.stereo_calibrate(mtx1, dist1, mtx2, dist2, 'synched/*')

    # this call might cause segmentation fault error. This is due to calling cv.imshow() and plt.show()
    #tools.triangulate(mtx1, mtx2, R, T)

if __name__ == "__main__":
    main()