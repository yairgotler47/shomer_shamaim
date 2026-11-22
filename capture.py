import cv2


def read_stereo_frames(cap_left, cap_right):
    grabbed_left = cap_left.grab()
    grabbed_right = cap_right.grab()
    if not (grabbed_left and grabbed_right):
        return False, None, False, None

    ret_left, frame_left = cap_left.retrieve()
    ret_right, frame_right = cap_right.retrieve()
    return ret_left, frame_left, ret_right, frame_right


def take_a_picture(cam):
    ret, frame = cam.read()
    if ret:
        cv2.imshow("Image", frame)
        cv2.waitKey(0)
        cv2.destroyWindow("Image")
    else:
        print("failed image capture")

    return ret, frame


def take_dual_picture(cap_left, cap_right):
    while True:
        ret_left, frame_left, ret_right, frame_right = read_stereo_frames(
            cap_left, cap_right
        )

        if ret_left and ret_right:
            cv2.imshow("Left Captured", frame_left)
            cv2.imshow("Right Captured", frame_right)
            if cv2.waitKey(1) != -1:
                print("captured")
                break
        else:
            print("Failed to capture stereo image pair.")

    return ret_left, frame_left, ret_right, frame_right


def take_n_pictures(camera, n, folder):
    images = []
    for _ in range(n):
        ret, frame = take_a_picture(camera)
        if ret:
            images.append(frame)
    return images


def take_n_dual_pictures(cap_left, cap_right, n):
    left_images = []
    right_images = []
    for _ in range(n):
        ret_left, frame_left, ret_right, frame_right = take_dual_picture(
            cap_left, cap_right
        )
        if ret_left and ret_right:
            left_images.append(frame_left)
            right_images.append(frame_right)
    return left_images, right_images