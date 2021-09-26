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

def edge_gradient_internal(img, k=3, iterations=1):
    """
    Normalized difference of an image from its morphological erosion.

    Parameters
    ----------
    img : ndarray
        Input single-channel image.
    k : int, default=3
        Aperture size of the square erosion kernel.
    iterations : int, default=1
        Number of successive erosion operations performed on `img`.

    Returns
    -------
    img_edge : ndarray
        Normalised, between (0, 255), difference between `img` and its
        erosion.
        Proportional to the internal gradient.

    """
    kernel = np.ones((k, k))
    img_erode = cv2.erode(img, kernel=kernel, iterations=iterations)
    img_edge = norm(diff(img, img_erode))
    return img_edge

def edge_gradient_external(img, k=3, iterations=1):
    """
    Normalized difference of an image from its morphological dilation.

    Parameters
    ----------
    img : ndarray
        Input single-channel image.
    k : int, default=3
        Aperture size of the square dilation kernel.
    iterations : int, default=1
        Number of successive dilation operations performed on `img`.

    Returns
    -------
    img_edge : ndarray
        Normalised, between (0, 255), difference between `img` and its
        dilation, inverted between (0, 255).
        Proportional to the external gradient.

    """
    kernel = np.ones((k, k))
    img_dilate = cv2.dilate(img, kernel=kernel, iterations=iterations)
    img_edge = invert(norm(diff(img, img_dilate)))
    return img_edge

def edge_gradient(img, k=3, iterations=1):
    """
    Normalized morphological gradient of an iamge.

    Parameters
    ----------
    img : ndarray
        Input single-channel image.
    k : int, default=3
        Aperture size of the square dilation kernel.
    iterations : int, default=1
        Number of successive gradient operations performed on `img`.

    Returns
    -------
    img_edge : ndarray
        Normalised, between (0, 255), morphological gradient of `img`.

    """
    kernel = np.ones((k, k))
    img_gradient = cv2.morphologyEx(img, op=cv2.MORPH_GRADIENT, kernel=kernel,
                                    iterations=iterations)
    img_edge = invert(norm(img_gradient))
    return img_edge
