#!/usr/bin/env python

import os
import argparse
import numpy as np
import cv2

from image_primitive import *
from image_edge import *
from detect import *


def detection(img):
    return


def recognition(img):
    return


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True,
                        help="directory path with input images")
    parser.add_argument("-w", "--work", required=True,
                        help="directory path for work images")
    parser.add_argument("-o", "--output", required=True,
                        help="directory path for output images and data")
    parser.add_argument("-W", "--work-save", action="store_true",
                        help="flag if work images are to be saved")

    args = vars(parser.parse_args())

    dir_input = args["input"]

    img_files = [os.path.join(dir_input, f)
                 for f in os.listdir(dir_input)
                 if os.path.isfile(os.path.join(dir_input, f))
                 and os.path.splitext(f)[1] in {".jpg", ".png"}]

    return args, img_files


# building signage
args, img_files = parse_input()
dir_work = args["work"]

for img_file in img_files:
    root, ext = os.path.splitext(os.path.basename(img_file))
    print(f"{img_file} -> ({root}, {ext})")

    def write_to_work(id, img):
        if args["work_save"]:
            cv2.imwrite(f"{dir_work}/{root}_{id}{ext}", img)
        return

    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if img.size == 0:
        print(f"{img_file} could not be opened")
        continue
    W, H = img.shape[:2]

    write_to_work("0_0", img)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    write_to_work("0_1", img_gray)

    # development below

    # parameters
    blur_k = 21
    blur_sigma_c = 50
    blur_sigma_s = 50

    edge_k = 3
    edge_iterations = 2

    edge_bin_k = 15
    edge_bin_c = -50

    # expand white character region into black background
    img_blur = cv2.bilateralFilter(img_gray, blur_k, blur_sigma_c, blur_sigma_s)
    write_to_work("1_0", img_blur)

    img_edge = edge_gradient_external(img_blur, edge_k, edge_iterations)
    write_to_work("1_1", img_edge)

    img_edge_bin = binarize(img_edge, edge_bin_k, edge_bin_c)
    write_to_work("1_2", img_edge_bin)

    img_ccl = detect_ccl(img_edge_bin)
    write_to_work("1_3", img_ccl)

    # contract black background around white characters
    img_blur = cv2.bilateralFilter(img_gray, blur_k, blur_sigma_c, blur_sigma_s)
    write_to_work("2_0", img_blur)

    img_edge = edge_gradient_internal(img_blur, edge_k, edge_iterations)
    write_to_work("2_1", img_edge)

    img_edge_bin = invert(binarize(img_edge, edge_bin_k, edge_bin_c))
    write_to_work("2_2", img_edge_bin)

    img_ccl = detect_ccl(img_edge_bin)
    write_to_work("2_3", img_ccl)

    # using regions rather than edges
    img_blur = cv2.bilateralFilter(img_gray, blur_k, blur_sigma_c, blur_sigma_s)
    write_to_work("3_0", img_blur)

    img_bin = binarize(img_blur, edge_bin_k, edge_bin_c)
    write_to_work("3_1", img_bin)

    img_ccl = detect_ccl(img_bin)
    write_to_work("3_2", img_ccl)
