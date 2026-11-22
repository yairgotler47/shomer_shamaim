import os

import cv2
import numpy as np

import calibrate
import tools
import trajectory
import triangulate


def sharpen_image(image):
    rows, cols = image.shape[:2]

    # gaussian kernel for sharpening
    gaussian_blur = cv2.GaussianBlur(image,(7,7),sigmaX=2)

    # sharpening using addWeighted()
    sharp1 = cv2.addWeighted(image,1.5,gaussian_blur,-0.5,0)
    sharp2 = cv2.addWeighted(image,3.5,gaussian_blur,-2.5,0)
    sharp3 = cv2.addWeighted(image,7.5,gaussian_blur,-6.5,0)
    # the addweight() method performs a linear comnbination of matrices
    # which is simple arithmetic operations for example the first function will
    # have the resultant matrix as image * 1.5 + gaussian_blur * (-0.5) + 0

    # showing the images
    # cv.imshow('sharp1',sharp1)
    # cv.imshow('sharp2',sharp2)
    # cv.imshow('sharp3',sharp3)
    # cv.imshow('original',image)
    # cv.waitKey(0)
    return sharp1

# object detection
def find_object_center(img, winname):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # values from find_hsv.py
    lower = np.array([17, 98, 118])
    upper = np.array([179, 197, 236])

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius > 5:
            center = (int(x), int(y))
            cv2.circle(img, center, int(radius), (0, 255, 0), 2)
            cv2.imshow(winname, img)
            cv2.waitKey(1)
            return x, y, img
    cv2.imshow(winname, img)
    cv2.waitKey(1)
    return None, None,None


def process_video(cam):
    points = []
    images = []
    ret, background = cam.read()
    cv2.imshow("background", background)
    cv2.waitKey(0)
    cv2.destroyWindow("background")

    count = 0
    while ret:
        ret, frame = cam.read()
        if not (ret):
            break
        x,y,img = find_object_center(frame, "image")
        point = (x,y)
        print(point)
        if x is not None and y is not None:
            points.append(point)
        if img is not None:
            images.append(img)

        count += 1
        if count == 100:
            break

    return points, images

def save_as_images(images, folder="saved_frames"):
    valid_images = [img for img in images if img is not None]
    if not valid_images:
        print("No images in the folder")
        return

    os.makedirs(folder, exist_ok=True)
    for idx, img in enumerate(valid_images):
        filepath = os.path.join(folder, f"frame_{idx:03d}.png")
        cv2.imwrite(filepath, img)
    print(f"Saved {len(valid_images)} images to {folder}")

if '__main__' == __name__:
    cam = cv2.VideoCapture(0)
    points, images = process_video(cam)
    print(points)

    for img in images:
        cv2.imshow("Slideshow", img)
        if cv2.waitKey(2000) & 0xFF == ord("q"):
            break

    save_as_images(images)

    cv2.destroyAllWindows()

    #tools.show_2d_graph(points)