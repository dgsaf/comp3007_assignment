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

    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if img.size == 0:
        print(f"{img_file} could not be opened")
        continue

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # write image and grayscale image to work directory
    cv2.imwrite(f"{dir_work}/{root}_0{ext}", img)

    cv2.imwrite(f"{dir_work}/{root}_1_0{ext}", img_gray)

    # development below
    img_edge = edge_gradient_external(img_gray)
    cv2.imwrite(f"{dir_work}/{root}_2_0{ext}", img_edge)
