#!/usr/bin/env python

import os
import argparse
import numpy as np
import cv2
import random

from image_primitive import *
from image_edge import *
from image_region import *
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
    if img is None:
        print(f"{img_file} could not be opened")
        continue
    W, H = img.shape[:2]

    write_to_work("0", img)

    # development below
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    write_to_work("1", img_gray)

    mser = cv2.MSER_create()
    mser.setMinArea(25)
    mser.setMaxArea(2000)
    mser.setDelta(25)
    point_sets, boxes = mser.detectRegions(img_gray)

    img_mser = np.zeros(img.shape)
    for ps in point_sets:
        color = (random.randint(100, 255),
                 random.randint(100, 255),
                 random.randint(100, 255))
        for p in ps:
            j, i = tuple(p[:2])
            img_mser[i, j] = color
    write_to_work("2", img_mser)

    regions = []
    for (ps, b) in zip(point_sets, boxes):
        regions.append(Region(ps, b))

    print(f"{len(regions)}")
    for i in range(len(regions)):
        regions[i].display()
        for j in range(len(regions)):
            f = regions[i].overlap(regions[j])
            if (f > 0.9):
                print(f"{i} ~contains {j}")

    # fs_blur = [
    #     lambda img_gray: cv2.bilateralFilter(img_gray, 11, 50, 25)
    # ]
    # fs_edge = [
    #     # lambda img: edge_morph_external(img),
    #     # lambda img: edge_morph_internal(img),
    #     # lambda img: edge_morph(img),
    #     # lambda img: edge_sobel(img),
    #     # lambda img: edge_scharr(img),
    #     # lambda img: edge_laplacian(img),
    #     # lambda img: edge_difference_gaussian(img, 1.0, 5.0),
    #     lambda img: edge_canny(img, t_1=25, t_2=250)
    #     # lambda img: edge_canny(img, t_1=25, t_2=350),
    #     # lambda img: edge_canny(img, t_1=25, t_2=450)
    # ]
    # fs_bin = [
    #     lambda img_gray: binarize(img_gray, k=31, c=-15)
    # ]

    # i = 1
    # for f_blur in fs_blur:
    #     j = 1
    #     for f_edge in fs_edge:
    #         k = 1
    #         for f_bin in fs_bin:
    #             img_gray, img_blur, img_edge, img_edge_bin = detect_edges(
    #                 img, f_blur, f_edge, f_bin)
    #             write_to_work(f"{i}_{j}_{k}_1", img_gray)
    #             write_to_work(f"{i}_{j}_{k}_2", img_blur)
    #             write_to_work(f"{i}_{j}_{k}_3", img_edge)
    #             write_to_work(f"{i}_{j}_{k}_4", img_edge_bin)

    #             img_contours = detect_digits(img_edge_bin)
    #             write_to_work(f"{i}_{j}_{k}_5", img_contours)

    #             k += 1
    #         j += 1
    #     i += 1
