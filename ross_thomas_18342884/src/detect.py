#!/usr/bin/env python

import os
import numpy as np
import cv2

def background(img, n, k):
    img_bg = img.copy()
    for i in range(n):
        # note: must have k - 2*n >= 1
        img_bg = cv2.medianBlur(img_bg, k - (2*i))
    return img_bg

def salient(img, n, k):
    img_bg = background(img, n, k)
    img_fg = cv2.absdiff(img, img_bg)

    img_fg_norm = img_fg.copy()
    cv2.normalise(img_fg, img_fg_norm, alpha=0, beta=255,
                  norm_type=cv2.NORM_MINMAX)
    return img_fg_norm
