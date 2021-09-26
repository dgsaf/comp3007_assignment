#!/usr/bin/env python

import os
import numpy as np
import cv2
import random

from image_primitive import *

def detect_contours(args, img_file, img_gray):
    dir_work = args["work"]
    root, ext = os.path.splitext(os.path.basename(img_file))

    # development below
    img_blur = cv2.bilateralFilter(img_gray, 11, 50, 100)
    cv2.imwrite(f"{dir_work}/{root}_contours_1{ext}", img_blur)

    img_edge = edge_gradient_external(img_blur)
    cv2.imwrite(f"{dir_work}/{root}_contours_2{ext}", img_edge)

    img_edge_bin = binarize(img_edge)
    cv2.imwrite(f"{dir_work}/{root}_contours_3{ext}", img_edge_bin)

    _, contours, hierarchy = cv2.findContours(
        img_edge_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def aspect(contour):
        _, _, w, h = cv2.boundingRect(contour)
        return (w / h)

    def fill(contour):
        _, _, w, h = cv2.boundingRect(contour)
        return (cv2.contourArea(contour) / (w * h))

    def box_area(contour):
        _, _, w, h = cv2.boundingRect(contour)
        return (w * h)

    def within(x, ival):
        return ((ival[0] <= x) and (x <= ival[1]))

    img_contours = np.zeros(img.shape, dtype=np.uint8)
    for i in range(len(contours)):
        contour = contours[i]

        if (within(fill(contour), (0.1, 1.0))
            and within(aspect(contour), (0.2, 1.2))
            and within(box_area(contour) / (W * H), (0.0001, 0.5))):
            cv2.drawContours(img_contours, contours, i, (255, 255, 255),
                             hierarchy=hierarchy, maxLevel=0)
    cv2.imwrite(f"{dir_work}/{root}_contours_4{ext}", img_contours)
    return


def detect_ccl(args, img_file, img_gray):
    dir_work = args["work"]
    root, ext = os.path.splitext(os.path.basename(img_file))

    # development below
    img_blur = cv2.bilateralFilter(img_gray, 11, 50, 100)
    cv2.imwrite(f"{dir_work}/{root}_ccl_1{ext}", img_blur)

    img_edge = edge_gradient_external(img_blur)
    cv2.imwrite(f"{dir_work}/{root}_ccl_2{ext}", img_edge)

    img_edge_bin = binarize(img_edge)
    cv2.imwrite(f"{dir_work}/{root}_ccl_3{ext}", img_edge_bin)

    n, labels = cv2.connectedComponents(invert(img_edge_bin), 8)

    color = np.zeros((n, 3), dtype=np.uint8)
    for k in range(1, n):
        for c in range(3):
            color[k, c] = random.randint(0, 255)

    h, w = img_gray.shape
    img_ccl = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            img_ccl[i, j] = color[labels[i, j]]
    cv2.imwrite(f"{dir_work}/{root}_ccl_4{ext}", img_contours)
    return
