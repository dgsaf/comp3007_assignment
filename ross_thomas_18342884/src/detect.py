#!/usr/bin/env python

import os
import numpy as np
import cv2

def background(img, k, iterations):
    img_bg = img.copy()
    for i in range(iterations):
        img_bg = cv2.medianBlur(img_bg, k)
    return img_bg

def salient(img, k, iterations):
    img_bg = background(img, k, iterations)
    img_fg = cv2.absdiff(img, img_bg)

    img_fg_norm = img_fg.copy()
    cv2.normalize(img_fg, img_fg_norm, alpha=0, beta=255,
                  norm_type=cv2.NORM_MINMAX)
    return img_fg_norm
