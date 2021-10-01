#!/usr/bin/env python

import os
import numpy as np
import cv2

from image_primitive import diff, invert, norm


def edge_morph_internal(img_gray, k=3, iterations=1):
    kernel = np.ones((k, k))
    img_erode = cv2.erode(img_gray, kernel=kernel, iterations=iterations)
    img_edge = norm(diff(img_gray, img_erode))
    return img_edge


def edge_morph_external(img_gray, k=3, iterations=2):
    kernel = np.ones((k, k))
    img_dilate = cv2.dilate(img_gray, kernel=kernel, iterations=iterations)
    img_edge = norm(diff(img_gray, img_dilate))
    return img_edge


def edge_morph(img_gray, k=3, iterations=1):
    kernel = np.ones((k, k))
    img_gradient = cv2.morphologyEx(
        img_gray, op=cv2.MORPH_GRADIENT, kernel=kernel, iterations=iterations)
    img_edge = norm(img_gradient)
    return img_edge


def edge_sobel(img_gray):
    img_gradient_x = cv2.Sobel(img_gray, -1, 1, 0)
    img_gradient_y = cv2.Sobel(img_gray, -1, 0, 1)
    img_edge = norm(np.abs(img_gradient_x) + np.abs(img_gradient_y))
    return img_edge


def edge_scharr(img_gray):
    img_gradient_x = cv2.Scharr(img_gray, -1, 1, 0)
    img_gradient_y = cv2.Scharr(img_gray, -1, 0, 1)
    img_edge = norm(np.abs(img_gradient_x) + np.abs(img_gradient_y))
    return img_edge


def edge_laplacian(img_gray, ksize=3):
    img_edge = norm(np.abs(cv2.Laplacian(img_gray, -1, ksize=ksize)))
    return img_edge


def edge_difference_gaussian(img_gray, sigma_1, sigma_2, ksize=(3, 3)):
    img_gauss_1 = cv2.GaussianBlur(img_gray, ksize, sigma_1)
    img_gauss_2 = cv2.GaussianBlur(img_gray, ksize, sigma_2)
    img_edge = norm(np.abs(img_gauss_1 - img_gauss_2))
    return img_edge


def edge_canny(img_gray, t_1=100, t_2=200, k=3):
    img_edge_bin = cv2.Canny(img_gray, t_1, t_2, apertureSize=k, L2gradient=True)
    return img_edge_bin
