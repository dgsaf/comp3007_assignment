#!/usr/bin/env python

import numpy as np
import cv2
import random

from parser import *
from region import *


# task 1
args, img_files = parse_input()

for img_file in img_files:
    file_root, file_ext, file_id = parse_image_file(img_file)
    print(f"{img_file} -> ({file_root}, {file_ext}, {file_id})")

    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if img is None:
        print(f"{img_file} could not be opened")
        continue
    W, H = img.shape[:2]

    write_to_work(args, img_file, "0", img)

    # development below
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    write_to_work(args, img_file, "1", img_gray)

    def write_regions_to_work(suffix, regions):
        print(f"number of regions = {len(regions)}")
        img_regions = np.zeros(img.shape)
        for region in regions:
            color = (random.randint(100, 255),
                     random.randint(100, 255),
                     random.randint(100, 255))
            for point in region.points:
                j, i = point
                img_regions[i, j] = color
            write_to_work(args, img_file, suffix, img_regions)

    regions = mser_regions(
        img_gray, min_area=25, max_area=2000, delta=20, threshold=0.8)
    write_regions_to_work("2_0_unique", regions)

    regions = list(filter(lambda r: r.box.aspect >= 0.8, regions))
    write_regions_to_work("2_1_aspect", regions)

    for i in range(len(regions)):
        print(str(regions[i]))
