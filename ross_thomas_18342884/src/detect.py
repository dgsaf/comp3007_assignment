#!/usr/bin/env python

import os
import numpy as np
import cv2
import random

from image_primitive import *
from image_edge import *


def detect_edges(img, f_blur, f_edge, f_bin):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = f_blur(img_gray)
    img_edge = f_edge(img_blur)
    img_edge_bin = f_bin(img_edge)
    return img_gray, img_blur, img_edge, img_edge_bin


def filter_contour(contour):
    l = cv2.arcLength(contour, closed=True)
    a = cv2.contourArea(contour, oriented=False)
    x, y, w, h = cv2.boundingRect(contour)

    valid_ratio = (0.15*l <= a <= 0.95*l)
    valid_aspect = (0.85*h <= w <= 2.55)
    valid_fill = (0.15*(h*w) <= a <= 0.85*(h*w))

    return (valid_ratio and valid_aspect and valid_fill)


def filter_adjacent(contour_1, contour_2):
    x_1, y_1, w_1, h_1 = cv2.boundingRect(contour_1)
    x_2, y_2, w_2, h_2 = cv2.boundingRect(contour_2)

    # non-adjacent if one contour contains the other wholly
    contained = ((x_1 <= x_2 <= x_1 + w_1 - w_2
                  and y_1 <= y_2 <= y_1 + h_1 - h_2)
                 or (x_2 <= x_1 <= x_2 + w_2 - w_1
                     and y_2 <= y_1 <= y_2 + h_2 - h_1))

    # non-adjacent if not of similar size
    similar_height = (0.5*h_2 <= h_1 <= (1/0.5)*h_2)
    similar_width = (0.5*w_2 <= w_1 <= (1/0.5)*w_2)

    # scale-independent adjacency
    dist_x = np.amin(np.abs(np.array([
        x_1 - x_2,
        x_1 + w_1 - x_2,
        x_1 - x_2 - w_2,
        x_1 + w_1 - x_2 - w_2])))

    dist_y = np.amin(np.abs(np.array([
        y_1 - y_2,
        y_1 + h_1 - y_2,
        y_1 - y_2 - h_2,
        y_1 + h_1 - y_2 - h_2])))

    neighbouring_x = (dist_x <= 4*w_1 and dist_x <= 4*w_2)
    neighbouring_y = (dist_y <= h_1 and dist_y <= h_2)

    return ((not contained)
            and similar_height and similar_width
            and neighbouring_x and neighbouring_y)


def centre(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return (int(x + 0.5*w), int(y + 0.5*h))


def detect_digits(img_edge_bin):
    _, contours, hierarchy = cv2.findContours(
        img_edge_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    indexes = np.array(
        [i for i in range(len(contours)) if filter_contour(contours[i])])

    img_contours = np.zeros(img_edge_bin.shape)

    for i in range(len(contours)):
        cv2.drawContours(img_contours, contours, i, 100)

    for i in indexes:
        cv2.drawContours(img_contours, contours, i, 255)

        for j in indexes:
            if filter_adjacent(contours[i], contours[j]):
                cv2.line(img_contours, centre(contours[i]), centre(contours[j]), 200)

    return img_contours
