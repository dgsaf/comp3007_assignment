#!/usr/bin/env python

import os
import numpy as np
import cv2

from image_primitive import *

def region_median_blur(img, k=21, iterations=5):
    """
    Normalized difference of an image from its median-blurred background.

    Parameters
    ----------
    img : ndarray
        Input single-channel image.
    k : int, default=21
        Aperture size of the median-blur kernel.
    iterations : int, default=5
        Number of successive median-blur operations performed on `img`.

    Returns
    -------
    img_region : ndarray
        Normalised, between (0, 255), difference between `img` and its
        median-blurred background.

    """
    img_bg = img.copy()
    for i in range(iterations):
        img_bg = cv2.medianBlur(img_bg, ksize=k)
    img_region = norm(diff(img, img_bg))
    return img_region
