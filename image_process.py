import cv2
from cvzone.ColorModule import ColorFinder
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
def find_object_center(img,winname):

    # Webcam or video of a basketball shot

    myColorFinder = ColorFinder(False)
    hsvVals = {'hmin':9, 'smin': 28, 'vmin':0, 'hmax': 100, 'smax':247, 'vmax':248}

    imgColor, mask =myColorFinder.update(img, hsvVals)
    img = cv2.resize(img, (1280, 720))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Orange basketball mask (tune for your lighting)
    lower = np.array([10, 100, 100])
    upper = np.array([100, 200, 200])
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
            cv2.waitKey(2)
            return x,y
    cv2.imshow(winname,img)
    return