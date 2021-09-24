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
    parser.add_argument("dir")
    args = parser.parse_args()

    dir = args["dir"]
    img_files = [f for f in os.listdir(dir) if os.isfile(f)]

    return img_files

def main():
    img_files = read_input()

    for img_file in img_files:
        img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        root, ext = os.path.splitext(img_file)
        print(f"{img_file} -> ({root}, {ext})")
