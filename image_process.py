import cv2
import numpy as np

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
    lower = np.array([9, 28, 0])
    upper = np.array([100, 247, 248])

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
            return x, y
    cv2.imshow(winname, img)
    cv2.waitKey(1)
    return None