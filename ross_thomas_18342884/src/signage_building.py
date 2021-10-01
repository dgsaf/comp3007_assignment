#!/usr/bin/env python

import os
import argparse
import numpy as np
import cv2
import random as rng

from image_primitive import *
from image_edge import *
from detect import *


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

    write_to_work("0", img)

    # development below
    f_blur = lambda img_gray: cv2.bilateralFilter(img_gray, 21, 50, 50)
    fs_edge = [
        lambda img: edge_morph_external(img),
        lambda img: edge_morph_internal(img),
        lambda img: edge_morph(img),
        lambda img: edge_sobel(img),
        lambda img: edge_scharr(img),
        lambda img: edge_laplacian(img),
        lambda img: edge_difference_gaussian(img, 3.0, 6.0),
        lambda img: edge_canny(img)
    ]
    f_bin = lambda img_gray: binarize(img_gray, k=31, c=-15)

    i = 1
    for f_edge in fs_edge:
        img_gray, img_blur, img_edge, img_edge_bin = detect_edges(
            img, f_blur, f_edge, f_bin)

        write_to_work(f"{i}_1", img_gray)
        write_to_work(f"{i}_2", img_blur)
        write_to_work(f"{i}_3", img_edge)
        write_to_work(f"{i}_4", img_edge_bin)

        i += 1
