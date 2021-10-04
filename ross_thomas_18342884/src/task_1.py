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

    def write_regions_to_work(id, regions):
        print(f"number of regions = {len(regions)}")
        img_regions = np.zeros(img.shape)
        for region in regions:
            color = (random.randint(100, 255),
                     random.randint(100, 255),
                     random.randint(100, 255))
            for point in region.points:
                j, i = point
                img_regions[i, j] = color
        write_to_work(id, img_regions)

    regions = mser_regions(
        img_gray, min_area=25, max_area=2000, delta=20, threshold=0.8)
    write_regions_to_work("2_0_unique", regions)

    regions = list(filter(lambda r: r.aspect >= 0.8, regions))
    write_regions_to_work("2_1_aspect", regions)

    # for i in range(len(regions)):
    #     regions[i].display()
    #     for j in range(len(regions)):
    #         if i == j:
    #             continue

    #         f = regions[i].overlap(regions[j])
    #         if (f > 0.5):
    #             print(f"{i} contains {j}: {f}")
    #             regions[i].display()
    #             regions[j].display()

    #             cv2.imshow(f"{i}", regions[i].image())
    #             cv2.imshow(f"{j}", regions[i].image())
    #             cv2.moveWindow(f"{j}", 500, 50)
    #             cv2.waitKey(0)
    #             cv2.destroyAllWindows()
