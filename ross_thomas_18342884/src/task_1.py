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
        return f"{timer() - time_start:>.1f} s>"

    file_root, file_ext, file_id = parse_image_file(img_file)
    print(f"{img_file} -> ({file_root}, {file_ext}, {file_id})")

    print(f"{timing()} reading {img_file}")
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if img is None:
        print(f"{img_file} could not be opened")
        continue
    H, W = img.shape[:2]

    # write_image_to_work(args, img_file, "0", img)

    # development below
    print(f"{timing()} converting to grayscale")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # write_image_to_work(args, img_file, "1", img_gray)

    # # TASK 2: use morphology to improve MSER near edge of sign
    # print(f"{timing()} morphology")
    # img_bg = cv2.morphology(
    #     img_gray, cv2.MORPH_OPEN,
    #     cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 31)), iterations=1)
    # write_image_to_work(args, img_file, "1_1", img_bg)
    # img_gray = cv2.addWeighted(img_gray, 1, img_bg, -1, 0)
    # write_image_to_work(args, img_file, "1_2", img_gray)

    print(f"{timing()} calculating MSER")
    mser = cv2.MSER_create()
    mser.setMinArea(45)
    mser.setMaxArea(2000)
    mser.setDelta(20)
    point_sets, boxes = mser.detectRegions(img_gray)

    print(f"{timing()} constructing regions")
    regions = [Region(ps, b) for (ps, b) in zip(point_sets, boxes)]
    print(f"{timing()} writing regions ({len(regions)})")
    # write_image_to_work(args, img_file, "2_0", draw_regions(regions, (H, W)))

    print(f"{timing()} removing overlapping regions")
    regions = remove_overlapping(regions, max_overlap=0.8)
    print(f"{timing()} writing regions ({len(regions)})")
    # write_image_to_work(args, img_file, "2_1", draw_regions(regions, (H, W)))

    print(f"{timing()} filtering regions by aspect ratio")
    # 0.8 for directions, 1.2 for digits
    regions = list(filter(lambda r: 1.2 <= r.box.aspect <= 3.0, regions))
    print(f"{timing()} writing regions ({len(regions)})")
    # write_image_to_work(args, img_file, "2_2", draw_regions(regions, (H, W)))

    print(f"{timing()} removing occluded hole regions")
    regions = remove_occluded_holes(regions, max_boundary_distance=10)
    print(f"{timing()} writing regions ({len(regions)})")
    # write_image_to_work(args, img_file, "2_3", draw_regions(regions, (H, W)))

    print(f"{timing()} removing highly filled regions")
    regions = list(filter(lambda r: r.fill <= 0.85, regions))
    print(f"{timing()} writing regions ({len(regions)})")
    # write_image_to_work(args, img_file, "2_4", draw_regions(regions, (H, W)))

    print(f"{timing()} calculating chains of similar, adjacent regions")
    chains = find_chains(regions)

    chains = list(filter(lambda c: len(c) <= 3, chains))

    print(f"{timing()} writing chains ({len(chains)}) of regions ({len(regions)})")
    img_chains = draw_regions(regions, (H, W))
    for chain in chains:
        chain_box = covering_box([r.box for r in chain])
        if len(chain) <= 3:
            box_color = (255, 255, 255)
        else:
            box_color = (100, 100, 100)
        cv2.rectangle(img_chains, chain_box.tl, chain_box.br, box_color, 1)
    write_image_to_work(args, img_file, "3", img_chains)

    print(f"Chains >>>")
    for i, chain in enumerate(chains):
        # img_chain = draw_regions(chain)
        # write_image_to_work(args, img_file, f"3_{i}", img_chain)

        def summary(arr):
            return f"({np.amin(arr):.1f} | {np.average(arr):.1f} | {np.amax(arr):.1f})"

        print(f"Chain {i}:")
        print(f"height = {summary([r.box.height for r in chain])}")
        print(f"width = {summary([r.box.width for r in chain])}")
        print(f"aspect = {summary([r.box.aspect for r in chain])}")
        print(f"fill = {summary([r.fill for r in chain])}")
        print(f"otsu sep = {otsu_separation(img, covering_box([r.box for r in chain]))}")
        print(f"")
        for j, region in enumerate(chain):
            print(f"{i}-{j}: \n{str(region)}")

    print(f"{timing()} writing regions of interest")
    rois = [covering_box([r.box for r in c]) for c in chains]

    rois = merge_overlapping(rois, max_overlap=0.01)

    img_rois = img.copy()
    for i, roi in enumerate(rois):
        cv2.rectangle(img_rois, roi.tl, roi.br, (255, 255, 255), 1)
    write_image_to_work(args, img_file, f"4", img_rois)

    # for i, roi in enumerate(rois):
    #     img_roi = img[roi.indexes]
    #     write_image_to_work(args, img_file, f"4_{i}", img_roi)

    print(f"{timing()} ")
