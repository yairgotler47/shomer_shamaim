import cv2

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
