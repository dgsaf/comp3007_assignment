#!/usr/bin/env python

import numpy as np
import cv2
from timeit import default_timer as timer

from parser import *
from region import *
from chain import *


# task 1
args, img_files = parse_input()

for img_file in img_files:
    time_start = timer()

    def timing():
        return f"{timer() - time_start:>.1f}s>"

    file_root, file_ext, file_id = parse_image_file(img_file)
    print(f"{img_file} -> ({file_root}, {file_ext}, {file_id})")

    print(f"{timing()} reading {img_file}")
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if img is None:
        print(f"{img_file} could not be opened")
        continue
    H, W = img.shape[:2]

    write_image_to_work(args, img_file, "0", img)

    # development below
    print(f"{timing()} converting to grayscale")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    write_image_to_work(args, img_file, "1", img_gray)

    print(f"{timing()} calculating MSER")
    mser = cv2.MSER_create()
    mser.setMinArea(45)
    mser.setMaxArea(2000)
    mser.setDelta(15)
    point_sets, boxes = mser.detectRegions(img_gray)

    print(f"{timing()} constructing regions")
    regions = [Region(ps, b) for (ps, b) in zip(point_sets, boxes)]
    write_image_to_work(args, img_file, "2_0", draw_regions(regions, (H, W)))

    print(f"{timing()} removing overlapping regions")
    regions = remove_overlapping(regions, max_overlap=0.8)
    write_image_to_work(args, img_file, "2_1", draw_regions(regions, (H, W)))

    print(f"{timing()} filtering regions by aspect ratio")
    regions = list(filter(lambda r: r.box.aspect() >= 0.8, regions))
    write_image_to_work(args, img_file, "2_2", draw_regions(regions, (H, W)))

    print(f"{timing()} removing occluded hole regions")
    regions = remove_occluded_holes(regions, max_boundary_distance=10)
    write_image_to_work(args, img_file, "2_3", draw_regions(regions, (H, W)))

    for i in range(len(regions)):
        print(str(regions[i]))

    print(f"{timing()} calculating chains of similar, adjacent regions")
    chains = find_chains(regions)

    img_chains = draw_regions(regions, (H, W))
    for chain in chains:
        chain_box = covering_box([r.box for r in chain])
        cv2.rectangle(img_chains, chain_box.tl(), chain_box.br(),
                      (255,255,255), 1)
    write_image_to_work(args, img_file, "3", img_chains)

    print(f"{timing()} ")
