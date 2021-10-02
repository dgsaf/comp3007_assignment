#!/usr/bin/env python

import os
import argparse
import numpy as np
import cv2

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


def extract_features(contour):
    length = cv2.arcLength(contour, closed=True)
    area = cv2.contourArea(contour, oriented=False)
    x, y, w, h = cv2.boundingRect(contour)
    features = np.array([length, area, length / area, h, w, h / w])
    return features


# program
args, img_files = parse_input()
dir_work = args["work"]

samples = []
for img_file in img_files:
    root, ext = os.path.splitext(os.path.basename(img_file))
    print(f"{img_file} -> ({root}, {ext})")

    label, k = (root.split("_"))[0:2]
    print(f"{root} -> ({label}, {k})")

    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if img is None:
        print(f"{img_file} could not be opened")
        continue

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)

    samples.append((label, k, img_bin.copy()))

data = np.zeros((12, 5, 6), dtype=np.float32)
for sample in samples:
    label, k, img = sample

    # use cv2.RETR_EXTERNAL for only top-level contours
    test, contours, hierarchy = cv2.findContours(
        img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imwrite(f"{dir_work}/{label}_{k}.jpg", img)

    i = 0
    for contour in contours:
        features = extract_features(contour)
        print(f"{features}")

        img_contour = np.zeros(img.shape)
        cv2.drawContours(img_contour, contours, i, 255)
        cv2.imwrite(f"{dir_work}/{label}_{k}_{i}.jpg", img_contour)

        i += 1
