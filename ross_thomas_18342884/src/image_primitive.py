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


def binarize(img, k=15, c=-50):
    """
    Binarize a single-channel image, using adaptive thresholding.

    Parameters
    ----------
    img : ndarray
        Input single-channel image.
    k : int, default=15
        Aperture size for adaptive thresholding.
    c : int, default=-50
        Constant subtracted from weighted mean in adaptive thresholding.

    Returns
    -------
    img_bin : ndarray
        Binary image, with values in {0, 255}, obtained after thresholding.

    """
    img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, k, c)

    return img_bin
