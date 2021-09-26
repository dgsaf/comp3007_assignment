#!/usr/bin/env python

import os
import numpy as np
import cv2

def invert(img):
    """
    Invert a single-channel image between its min and max values.

    Parameters
    ----------
    img : ndarray
        Input single-channel images.

    Returns
    -------
    img_invert : ndarray
        Inverted copy of `img`.

    """
    a = np.amin(img)
    b = np.amax(img)

    img_invert = b + a - img
    return img_invert

def norm(img):
    """
    Normalize a single-channel image to have values between (0, 255).

    Parameters
    ----------
    img : ndarray
        Input single-channel images.

    Returns
    -------
    img_norm : ndarray
        Normalised, between (0, 255), copy of `img`.

    """
    img_norm = img.copy()
    cv2.normalize(img, img_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return img_norm

def diff(img_1, img_2):
    """
    Absolute difference of two single-channel images.

    Parameters
    ----------
    img_1, img_2 : ndarray
        Input single-channel images, with same numeric type.

    Returns
    -------
    img_diff : ndarray
        Absolute difference of the `img_1` and `img_2`.

    """
    img_diff = cv2.absdiff(img_1, img_2)
    return img_diff

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
