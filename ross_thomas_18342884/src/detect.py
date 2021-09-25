#!/usr/bin/env python

import os
import numpy as np
import cv2

def norm_diff(img_1, img_2):
    img_diff = cv2.absdiff(img_1, img_2)
    img_diff_norm = img_diff.copy()
    cv2.normalize(img_diff, img_diff_norm, alpha=0, beta=255,
                  norm_type=cv2.NORM_MINMAX)
    return img_diff_norm

def salient_median_blur(img, k=21, iterations=5):
    img_bg = img.copy()
    for i in range(iterations):
        img_bg = cv2.medianBlur(img_bg, ksize=k)
    img_salient = norm_diff(img, img_bg)
    return img_salient

def salient_erode(img, k=5, iterations=16):
    kernel = np.ones((k, k))
    img_bg = cv2.erode(img, kernel=kernel, iterations=iterations)
    img_salient = norm_diff(img, img_bg)
    return img_salient

def salient_dilate(img, k=3, iterations=16):
    kernel = np.ones((k, k))
    img_dilate = cv2.dilate(img, kernel=kernel, iterations=iterations)
    img_salient = 255 - norm_diff(img, img_dilate)
    return img_salient

def binarize_otsu(img):
    _, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_OTSU)
    return img_bin
