#!/usr/bin/env python

import os
import argparse
import numpy as np
import cv2

from image_primitive import *
from image_edge import *
from detect import *


def detection(img):
    return


def recognition(img):
    return


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True,
                        help="directory path with input images")
    parser.add_argument("-w", "--work", required=True,
                        help="directory path for work images")
    parser.add_argument("-o", "--output", required=True,
                        help="directory path for output images and data")

    args = vars(parser.parse_args())

    dir_input = args["input"]

    img_files = [os.path.join(dir_input, f)
                 for f in os.listdir(dir_input)
                 if os.path.isfile(os.path.join(dir_input, f))
                 and os.path.splitext(f)[1] in {".jpg", ".png"}]

    return args, img_files


# building signage
args, img_files = parse_input()
dir_work = args["work"]

for img_file in img_files:
    root, ext = os.path.splitext(os.path.basename(img_file))
    print(f"{img_file} -> ({root}, {ext})")

    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if img.size == 0:
        print(f"{img_file} could not be opened")
        continue

    cv2.imwrite(f"{dir_work}/{root}_0{ext}", img)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{dir_work}/{root}_1{ext}", img_gray)

    # development below
    img_blur = cv2.bilateralFilter(img_gray, 11, 50, 100)
    cv2.imwrite(f"{dir_work}/{root}_1_1{ext}", img_blur)

    img_edge = edge_gradient_external(img_blur)
    cv2.imwrite(f"{dir_work}/{root}_2{ext}", img_edge)

    img_edge_bin = binarize(img_edge)
    cv2.imwrite(f"{dir_work}/{root}_3{ext}", img_edge_bin)

    _, contours, hierarchy = cv2.findContours(
        img_edge_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def suitable_contour(contour):
        A = img.shape[0] * img.shape[1]

        a = (0.3, 1.0)
        f = (0.0001, 0.5)
        fr = (0.1, 1.0)

        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        aspect = w / h
        fill = w * h / A
        fill_rect = area / w * h

        within_aspect = (a[0] <= aspect) and (aspect <= a[1])
        within_fill = (f[0] <= fill) and (fill <= f[1])
        within_fill_rect = (fr[0] <= fill_rect) and (fill_rect <= fr[1])

        return (within_aspect and within_fill and within_fill_rect)


    img_contours = np.zeros(img.shape, dtype=np.uint8)
    for i in range(len(contours)):
        contour = contours[i]

        if (suitable_contour(contour)):
            cv2.drawContours(img_contours, contours, i, (255, 255, 255),
                             hierarchy=hierarchy, maxLevel=0)

    cv2.imwrite(f"{dir_work}/{root}_4{ext}", img_contours)
