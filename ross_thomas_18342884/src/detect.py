#!/usr/bin/env python

import os
import numpy as np
import cv2
import random

from image_primitive import *
from image_edge import *


def detect_ccl(img_edge_bin):
    n, labels = cv2.connectedComponents(img_edge_bin, 8)

    color = np.zeros((n, 3), dtype=np.uint8)
    for k in range(1, n):
        for c in range(3):
            color[k, c] = random.randint(0, 255)

    h, w = img_edge_bin.shape
    img_ccl = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            img_ccl[i, j] = color[labels[i, j]]
    return img_ccl


def binary_gradient_internal(img):
    blur_k = 21
    blur_sigma_c = 50
    blur_sigma_s = 50

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
    blur_sigma_c = 50
    blur_sigma_s = 50

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
    blur_sigma_c = 50
    blur_sigma_s = 50

    edge_k = 3
    edge_iterations = 2

    bin_k = 31
    bin_c = -15

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.bilateralFilter(img_gray, blur_k, blur_sigma_c, blur_sigma_s)
    img_bin = binarize(img_blur, bin_k, bin_c)
    return img_bin
