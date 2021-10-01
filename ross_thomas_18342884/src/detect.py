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

def detect_digits(img_edge_bin, f_filter_contours):
    _, contours, hierarchy = cv2.findContours(
        img_edge_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    indexes, edges = f_filter_contours(contours, hierarchy)
    # todo
    return
