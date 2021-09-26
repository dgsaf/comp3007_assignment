#!/usr/bin/env python

import os
import numpy as np
import cv2

from image_primitive import diff, invert, norm


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
    Normalized morphological gradient of an image.

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


def edge_canny(img, t_1=100, t_2=200, k=3):
    """
    Detects edges in an image, using the Canny edge detector.

    Parameters
    ----------
    img : ndarray
        Input single-channel image.
    t_1 : int, default=100
        Lower threshold for the Canny edge detector.
    t_2 : int, default=200
        Upper threshold for the Canny edge detector.
    k : int, default=3
        Apeture size for the Canny edge detector

    Returns
    -------
    img_edge : ndarray
        Binary image of edges, with same shape as `img`.

    """
    img_edge = invert(cv2.Canny(img, t_1, t_2, apertureSize=k, L2gradient=True))
    return img_edge
