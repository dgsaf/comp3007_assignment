#!/usr/bin/env python

import os
import argparse
import numpy as np
import cv2

from detect import salient_median_blur, salient_erode, salient_dilate

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

    # development below
    cv2.imwrite(f"{dir_work}/{root}_0{ext}", img)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{dir_work}/{root}_1_0{ext}", img_gray)

    for i in [1, 2, 4, 8, 16]:
        img_median = salient_median_blur(img, iterations=i)
        cv2.imwrite(f"{dir_work}/{root}_1_1_{i}{ext}", img_median)

    for i in [1, 2, 4, 8, 16]:
        img_erode = salient_erode(img, iterations=i)
        cv2.imwrite(f"{dir_work}/{root}_1_2_{i}{ext}", img_erode)

    for i in [1, 2, 4, 8, 16]:
        img_dilate = salient_dilate(img, iterations=i)
        cv2.imwrite(f"{dir_work}/{root}_1_3_{i}{ext}", img_dilate)
