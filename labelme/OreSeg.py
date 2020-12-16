import numpy as np
import cv2
import os

def simp_seg(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        return None, 0, 0
    
    img_h, img_w = image.shape[0], image.shape[1]
    kernel_size = max(5, int(max(image.shape[0], image.shape[1]) / 100 + 1))
    smoothed = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0.0)
    hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    # S = cv2.GaussianBlur(S, (0, 0), 2.0)
    thresh = 60
    _, mask = cv2.threshold(H, thresh, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    area_thresh = max(image.shape[0], image.shape[1])
    for idx in range(len(contours)):
        area = cv2.contourArea(contours[idx])
        if (area < area_thresh):
            continue
        perimeter = cv2.arcLength(contours[idx], True)
        circularity = 4 * 3.14 * area / (perimeter * perimeter + 1.0e-6)
        if (circularity < 0.3):
            continue
        approx = cv2.approxPolyDP(contours[idx], perimeter * 0.008, True)
        valid_contours.append(approx.squeeze())
        
    return valid_contours, img_w, img_h

if __name__ == '__main__':
    image_dir = "E:\\Datasets\\MineralData\\Imageokshi-1"
    for (dirpath, dirnames, filenames) in os.walk(image_dir):
        for filename in filenames:
            polygons, W, H = simp_seg(os.path.join(dirpath, filename))
            if polygons is None:
                continue
            mask = np.zeros([H, W, 1], np.uint8)
            cv2.drawContours(mask, polygons, -1, 255, -1)
            cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
            cv2.imshow("mask", mask)
            cv2.waitKey(0)