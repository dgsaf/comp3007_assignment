#!/usr/bin/env python

import numpy as np
import cv2

from parser import *
from region import *
from chain import *


# task 1
args, img_files = parse_input()

for img_file in img_files:
    file_root, file_ext, file_id = parse_image_file(img_file)
    print(f"{img_file} -> ({file_root}, {file_ext}, {file_id})")

    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if img is None:
        print(f"{img_file} could not be opened")
        continue
    H, W = img.shape[:2]

    write_image_to_work(args, img_file, "0", img)

    # development below
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    write_image_to_work(args, img_file, "1", img_gray)

    mser = cv2.MSER_create()
    mser.setMinArea(45)
    mser.setMaxArea(2000)
    mser.setDelta(15)
    point_sets, boxes = mser.detectRegions(img_gray)

    regions = [Region(ps, b) for (ps, b) in zip(point_sets, boxes)]
    write_image_to_work(args, img_file, "2_0", draw_regions(regions, (H, W)))

    regions = remove_overlapping(regions, max_overlap=0.8)
    write_image_to_work(args, img_file, "2_1", draw_regions(regions, (H, W)))

    regions = list(filter(lambda r: r.box.aspect() >= 0.8, regions))
    write_image_to_work(args, img_file, "2_2", draw_regions(regions, (H, W)))

    regions = remove_occluded_holes(regions, max_boundary_distance=10)
    write_image_to_work(args, img_file, "2_3", draw_regions(regions, (H, W)))

    for i in range(len(regions)):
        print(str(regions[i]))

    chains = find_chains(regions)
    i = 0
    for chain in chains:
        if len(chain) <= 3:
            write_image_to_work(args, img_file, f"2_4_{i}",
                                draw_regions(chain, (H, W)))
        i += 1

    # i = 0
    # for chain in chains:
    #     img_chain = np.zeros(img_gray.shape)
    #     for r in chain:
    #         # x1, x2 = r.box.x, r.box.x + r.box.width
    #         # y1, y2 = r.box.y, r.box.y + r.box.height
    #         # img_chain[y1:y2, x1:x2] = r.image()
    #         img_chain[r.box.indexes()] = r.image()
    #     write_image_to_work(args, img_file, f"3_chain_{i}", img_chain)
    #     i += 1
