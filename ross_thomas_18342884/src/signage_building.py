#!/usr/bin/env python

import os
import argparse
import numpy as np
import cv2
import random as rng

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

    write_to_work("0", img)

    # development below
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    write_to_work("1", img_gray)

    img_bin = binary_region(img_gray)
    write_to_work("2", img_bin)

    mser = cv2.MSER_create()
    mser.setMinArea(25)
    regions, boxes = mser.detectRegions(img_bin)

    img_regions = np.zeros((W, H, 3), dtype=np.uint8)
    for region in regions:
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        for point in region:
            j, i = point
            img_regions[i, j] = color
    write_to_work("3", img_regions)

    # implement combination of MSER detectors for +, -, R, G, B
