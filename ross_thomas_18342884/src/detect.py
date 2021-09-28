#!/usr/bin/env python

import os
import numpy as np
import cv2
import random

from image_primitive import *
from image_edge import *


def binary_gradient_internal(img):
    blur_k = 21
    blur_sigma_c = 75
    blur_sigma_s = 75

    edge_k = 3
    edge_iterations = 5

    bin_k = 31
    bin_c = -15

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.bilateralFilter(img_gray, blur_k, blur_sigma_c, blur_sigma_s)
    img_edge = edge_gradient_internal(img_blur, edge_k, edge_iterations)
    img_bin = binarize(img_edge, bin_k, bin_c)
    return img_bin


def binary_gradient_external(img):
    blur_k = 21
    blur_sigma_c = 75
    blur_sigma_s = 75

    edge_k = 3
    edge_iterations = 2

    bin_k = 31
    bin_c = -15

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.bilateralFilter(img_gray, blur_k, blur_sigma_c, blur_sigma_s)
    img_edge = edge_gradient_external(img_blur, edge_k, edge_iterations)
    img_bin = invert(binarize(img_edge, bin_k, bin_c))
    return img_bin


def binary_region(img):
    blur_k = 21
    blur_sigma_c = 75
    blur_sigma_s = 75

    edge_k = 3
    edge_iterations = 2

    bin_k = 31
    bin_c = -15

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.bilateralFilter(img_gray, blur_k, blur_sigma_c, blur_sigma_s)
    img_bin = binarize(img_blur, bin_k, bin_c)
    return img_bin


def detect_ccl(img_bin):
    n, labels = cv2.connectedComponents(img_bin, 8)

    color = np.zeros((n, 3), dtype=np.uint8)
    for k in range(1, n):
        for c in range(3):
            color[k, c] = random.randint(0, 255)

    h, w = img_bin.shape
    img_ccl = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            img_ccl[i, j] = color[labels[i, j]]
    return img_ccl


def similar(c, x_1, x_2):
    r = min(x_1, x_2) / max(x_1, x_2)
    return ((1/c) <= r <= c)


def digit_candidates(contours, hierarchy):
    n = len(contours)

    centres = np.zeros((n, 2), dtype=np.float32)
    sizes = np.zeros((n, 2), dtype=np.float32)
    angles = np.zeros((n), dtype=np.float32)

    for i in range(n):
        contour = contours[i]
        centres[i], sizes[i], angles[i] = cv2.minAreaRect(contour)

    # valid aspect ratios
    idx_valid = np.nonzero(similar(4.0, sizes[i, 0], sizes[i, 1]))

    # for i in idx_valid:
    #     j = hierarchy[0][i][0]
    #     while (j != -1):
    #         sim_width = similar(4.0, sizes[i, 0], sizes[j, 0])
    #         sim_height = similar(2.0 , sizes[i, 1], sizes[j, 1])
    #         j = hierarchy[0][j][0]

    return idx_valid


def detect_contours(img_bin):
    _, contours, hierarchy = cv2.findContours(
        img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    n = len(contours)

    centres = np.zeros((n, 2), dtype=np.float32)
    sizes = np.zeros((n, 2), dtype=np.float32)
    angles = np.zeros((n), dtype=np.float32)

    for i in range(len(contours)):
        contour = contours[i]
        centres[i], sizes[i], angles[i] = cv2.minAreaRect(contour)

    H, W = img_bin.shape
    img_contours = np.zeros((H, W), dtype=np.uint8)
    for i in range(n):
        # cv2.circle(img_contours, centres[i], 3, 255)
        cv2.ellipse(img_contours, (centres[i], sizes[i], angles[i]), 128)

    idx_valid = digit_candidates(contours, hierarchy)
    for i in idx_valid:
        cv2.ellipse(img_contours, (centres[i], sizes[i], angles[i]), 255)

    return img_contours
