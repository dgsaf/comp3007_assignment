#!/usr/bin/env python

import os
import argparse
import numpy as np
import cv2

def detection(img):
    return

def recognition(img):
    return

def read_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True,
                        help="directory with input images")
    parser.add_argument("-w", "--work", required=True,
                        help="directory for work images")
    parser.add_argument("-o", "--output", required=True,
                        help="directory for output images and data")

    args = vars(parser.parse_args())

    dir_input = args["i"]

    files = [f for f in os.listdir(dir_input)
             if os.path.isfile(os.path.join(dir_input, f))]

    img_files = [os.path.join(dir_input, f)
                 for f in os.listdir(dir_input)
                 if os.path.isfile(os.path.join(dir_input, f))
                 and os.path.splitext(f)[1] == ".jpg"]

    return args, img_files

# program
args, img_files = read_input()

for img_file in img_files:
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    root, ext = os.path.splitext(os.path.basename(img_file))
    print(f"{img_file} -> ({root}, {ext})")
