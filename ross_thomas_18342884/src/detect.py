#!/usr/bin/env python

import os
import numpy as np
import cv2

def norm_diff(img_1, img_2):
    """
    Difference of two single-channel images, normalized between (0, 255).

    Parameters
    ----------
    img_1, img_2 : ndarray
        Input single-channel images, with same numeric type.

    Returns
    -------
    img_diff_norm : ndarray
        Normalised, between (0, 255), difference of the two input images.

    """
    img_diff = cv2.absdiff(img_1, img_2)
    img_diff_norm = img_diff.copy()
    cv2.normalize(img_diff, img_diff_norm, alpha=0, beta=255,
                  norm_type=cv2.NORM_MINMAX)
    return img_diff_norm

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
    img_region = norm_diff(img, img_bg)
    return img_region

def edge_erode(img, k=3, iterations=1):
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
        erosion; proportional to the internal gradient.

    """
    kernel = np.ones((k, k))
    img_bg = cv2.erode(img, kernel=kernel, iterations=iterations)
    img_edge = norm_diff(img, img_bg)
    return img_edge

def edge_dilate(img, k=3, iterations=1):
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
        dilation; proportional to the external gradient.

    """
    kernel = np.ones((k, k))
    img_dilate = cv2.dilate(img, kernel=kernel, iterations=iterations)
    img_edge = 255 - norm_diff(img, img_dilate)
    return img_edge

def edge_gradient(img, k=3, iterations=1):
    kernel = np.ones((k, k))
    img_gradient = cv2.morphologyEx(img, op=cv2.MORPH_GRADIENT, kernel=kernel,
                                    iterations=iterations)
    img_edge = img_gradient.copy()
    cv2.normalize(img_gradient, img_edge, alpha=0, beta=255,
                  norm_type=cv2.NORM_MINMAX)
    return img_edge

def binarize(img, threshold=None):
    """
    Binarize a single-channel image.

    Parameters
    ----------
    img : ndarray
        Input single-channel image.
    threshold : int, optional
        Threshold value for binary thresholding.
        Default is None, in which case Otsu's method is used.

    Returns
    -------
    img_bin : ndarray
        Binary image, with values in {0, 255}, obtained after thresholding.

    """
    if threshold is None:
        _, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_OTSU)
    else:
        _, img_bin = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    return img_bin

def preprocess(img, k_blur=3, k_dilate=3, iterations=10, threshold=None):
    img_blur = cv2.medianBlur(img, ksize=k_blur)
    img_dilate = salient_dilate(img_blur, k=k_dilate, iterations=iterations)
    img_bin = binarize(img, threshold)
    return img_bin
