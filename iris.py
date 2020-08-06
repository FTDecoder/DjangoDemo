import cv2
import numpy as np
import os

centroid = (0, 0)
radius = 0
current_eye = 0
eyes_list = []


def getNewEye(list):
    global current_eye
    if current_eye >= len(list):
        current_eye = 0
    new_eye = list[current_eye]
    current_eye += 1
    return new_eye


def getIris(frame):
    iris = []
    array = np.asarray(frame)
    copy_img = array.copy()
    res_img = array.copy()
    mask = np.zeros(array.shape, np.uint8)
    gray_img = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    canny_img = cv2.Canny(gray_img, 100, 250)
    smooth_img = cv2.GaussianBlur(canny_img, (7, 7), 1)
    circles = getCircles(smooth_img)
    iris.append(res_img)
    for circle in circles:
        rad = int(circle[0][2])
        global radius
        radius = rad
        cv2.circle(mask, centroid, rad, (255, 255, 255), 3, cv2.FILLED)
        inv = cv2.bitwise_not(mask, mask)
        cv2.subtract(array, copy_img, res_img, mask=inv)
        x = int(centroid[0] - rad)
        y = int(centroid[1] - rad)
        w = int(rad * 2)
        h = w
        cv2.circle(res_img, (x, y), (w, h), (0, 255, 0), 2)
        crop_img = np.asarray(array[w, h, 3], np.uint8, )
        cv2.copyTo(res_img, crop_img)
        return crop_img
    return res_img


def getCircles(image):
    i = 80
    while i < 151:
        # cv2.imread(image)
        array = np.asarray(image)
        storage = np.array(array.shape[1], np.uint8)
        cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 2, 100, i, minRadius=100, maxRadius=140)
        circles = np.asarray(storage)
        if circles is True:
            return circles
        i += 1
    return []


def getPupil(frame):
    array = np.asarray(frame)
    pupil_img = np.array(array.shape, np.uint8)
    cv2.inRange(frame, (30, 30, 30), (80, 80, 80), pupil_img)
    contours, hierarchy = cv2.findContours(pupil_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pupil_img1 = array.copy()
    while contours is True:
        moments = cv2.moments(contours)
        area = cv2.contourArea(contours)
        if area > 50:
            x = moments['m10'] / area
            y = moments['m01'] / area
            pupil = contours
            global centroid
            centroid = (int(x), int(y))
            cv2.drawContours(pupil_img1, pupil, -1, (255, 0, 0), 2, cv2.FILLED)
            break
    return pupil_img1


def getPolar2CartImg(image, rad):
    array = np.array(image)
    c = (float(array.shape[0] / 4.0), float(array.shape[1] / 4.0))
    # img_res = np.asarray([rad, int(360), 3], np.uint8)
    img = cv2.logPolar(array, c, 5, cv2.INTER_LINEAR)
    return img


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.21, 0.72, 0.07])



cv2.namedWindow("Input", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Output", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Normalized", cv2.WINDOW_AUTOSIZE)

List = os.listdir('image/eyes')
key = 0
while True:
    eye = getNewEye(List)
    new_frame = cv2.imread("image/eyes/" + eye)
	new_frame = cv2.resize(new_frame, (240,240))
    frame = np.array(new_frame)
    gray = rgb2gray(frame)
    gray /= 255
    iris1 = frame.copy()
    output = getPupil(frame)
    iris = getIris(output)
    norm_Img = getPola2CartImg(iris1, radius)
	cv2.imshow("iInput", frame)
    cv2.imshow("Output", gray)
    cv2.imshow("Normalized", norm_Img)
    key = cv2.waitKey(3000)
    if key == 27 or key == 1048603:
        break

cv2.destroyAllWindows()
