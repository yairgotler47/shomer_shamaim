import cv2
import tools
CAL_IMAGE_NUM=15
def capture_and_calibrate_camera(camera):
    return
def capture_and_calibrate_stereo(cap_left,cap_right):
    left_images,right_images=tools.take_n_dual_pictures(cap_left,cap_right,CAL_IMAGE_NUM)
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

def main():
    mtx1, dist1 = tools.calibrate_camera(images_folder='left_cal_images/*')
    mtx2, dist2 = tools.calibrate_camera(images_folder='right_cal_images/*')

    R, T = tools.stereo_calibrate(mtx1, dist1, mtx2, dist2, 'synched/*')

    # this call might cause segmentation fault error. This is due to calling cv.imshow() and plt.show()
    tools.triangulate(mtx1, mtx2, R, T)
