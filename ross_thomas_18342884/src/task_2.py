#!/usr/bin/env python3

import numpy as np
import cv2
from timeit import default_timer as timer

from parser import *
from region import *
from chain import *
from knn import *


# task 2
args, img_files = parse_input()

# build classifiers
print(f"> building classifiers")
knn_digits = build_knn_digits(args["digits"], 5, 7)
knn_arrows = build_knn_arrows(args["digits"], 3, 3)

# locate and classify the digits of each line of each directional sign
for img_file in img_files:
    time_img = timer()
    def timing():
        return f"{timer() - time_img:>.1f} s>"

    file_root, file_ext, file_id = parse_image_file(img_file)
    print(f"{img_file} -> ({file_root}, {file_ext}, {file_id})")

    def write_image_to_work(suffix, img_work):
        cv2.imwrite(f"{args['work']}/{file_root}_{suffix}{file_ext}", img_work)
        return

    print(f"{timing()} reading {img_file}")
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if img is None:
        print(f"{img_file} could not be opened")
        continue
    H, W = img.shape[:2]

    if args["work_save"]:
        print(f"{timing()} writing image to work")
        write_image_to_work("0", img)

    # development below
    print(f"{timing()} converting to grayscale")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if args["work_save"]:
        print(f"{timing()} writing grayscale image to work")
        write_image_to_work("1", img_gray)

    # # TASK 2: use morphology to improve MSER near edge of sign
    # print(f"{timing()} morphology")
    # img_bg = cv2.morphology(
    #     img_gray, cv2.MORPH_OPEN,
    #     cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 31)), iterations=1)
    # write_image_to_work("1_1", img_bg)
    # img_gray = cv2.addWeighted(img_gray, 1, img_bg, -1, 0)
    # write_image_to_work("1_2", img_gray)

    print(f"{timing()} calculating MSER")
    mser = cv2.MSER_create()
    mser.setMinArea(45)
    mser.setMaxArea(2000)
    mser.setDelta(20)
    point_sets, boxes = mser.detectRegions(img_gray)

    print(f"{timing()} constructing regions")
    regions = [Region(ps) for ps in point_sets]

    if args["work_save"]:
        print(f"{timing()} writing regions ({len(regions)})")
        write_image_to_work("2_0", draw_regions(regions, (H, W)))

    print(f"{timing()} removing overlapping regions")
    regions = remove_overlapping(regions, max_overlap=0.8)

    if args["work_save"]:
        print(f"{timing()} writing regions ({len(regions)})")
        write_image_to_work("2_1", draw_regions(regions, (H, W)))

    print(f"{timing()} filtering regions by aspect ratio")
    regions = list(filter(lambda r: 0.75 <= r.box.aspect <= 3.0, regions))

    if args["work_save"]:
        print(f"{timing()} writing regions ({len(regions)})")
        write_image_to_work("2_2", draw_regions(regions, (H, W)))

    print(f"{timing()} removing occluded hole regions")
    regions = remove_occluded_holes(regions, max_boundary_distance=10)

    if args["work_save"]:
        print(f"{timing()} writing regions ({len(regions)})")
        write_image_to_work("2_3", draw_regions(regions, (H, W)))

    print(f"{timing()} removing highly filled regions")
    regions = list(filter(lambda r: r.fill <= 0.85, regions))

    if args["work_save"]:
        print(f"{timing()} writing regions ({len(regions)})")
        write_image_to_work("2_4", draw_regions(regions, (H, W)))

    print(f"{timing()} calculating chains of similar, adjacent regions")
    chains = find_chains(regions)

    if args["work_save"]:
        print(f"{timing()} writing chains ({len(chains)})")
        img_chains = draw_regions(regions, (H, W))
        for chain in chains:
            chain_box = covering_box([r.box for r in chain])
            cv2.rectangle(
                img_chains, chain_box.tl, chain_box.br, (255, 255, 255), 1)
        write_image_to_work("3", img_chains)

    print(f"{timing()} filtering chains by length")
    chains = list(filter(lambda c: len(c) <= 3, chains))

    if not chains:
        print(f"{timing()} no suitable chains found")
        print(f"{timing()} ")
        continue

    if args["work_save"]:
        print(f"{timing()} writing regions of interest")
        rois = [covering_box([r.box for r in c]) for c in chains]
        rois = merge_overlapping(rois, max_overlap=0.01)

        img_rois = img.copy()
        for i, roi in enumerate(rois):
            cv2.rectangle(img_rois, roi.tl, roi.br, (255, 255, 255), 1)
        write_image_to_work("4", img_rois)

        for i, roi in enumerate(rois):
            img_roi = img[roi.indexes]
            write_image_to_work(f"4_{i}", img_roi)

    # print(f"{timing()} selecting chain most likely to be digits")
    # chain_digits = cluster_largest_otsu_separations(img, chains)[0]

    # if args["work_save"]:
    #     print(f"{timing()} writing digit chain")
    #     img_digits = img[covering_box([r.box for r in chain_digits]).indexes]
    #     write_image_to_work("5", img_digits)

    # print(f"{timing()} classifying digits")
    # features_digits = np.array(
    #     [np.ravel(r.spatial_histogram(5, 7))
    #      for r in chain_digits])
    # predicted_digits = knn_digits.predict(features_digits, k=3)

    # print(f"{timing()} writing output for {file_root}{file_ext}")
    # img_digits = img[covering_box([r.box for r in chain_digits]).indexes]
    # cv2.imwrite(f"{args['output']}/DetectedArea{file_id}{file_ext}", img_digits)

    # with open(f"{args['output']}/Building{file_id}.txt", "w") as out_file:
    #     str_digits = "".join(map(str, predicted_digits))
    #     print(f"Building {str_digits}", file=out_file)

    # print(f"{timing()} ")
