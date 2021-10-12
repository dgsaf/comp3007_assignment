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
knn_digits = build_knn_digits(args["digits"], 3, 5)
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
        # write_image_to_work("1", img_gray)

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
    mser.setMinArea(25)
    mser.setMaxArea(2000)
    mser.setDelta(20)
    point_sets, boxes = mser.detectRegions(img_gray)

    print(f"{timing()} constructing regions")
    regions = [Region(ps) for ps in point_sets]

    if args["work_save"]:
        print(f"{timing()} writing regions ({len(regions)})")
        # write_image_to_work("2_0", draw_regions(regions, (H, W)))

    print(f"{timing()} removing overlapping regions")
    regions = remove_overlapping(regions, max_overlap=0.8)

    if args["work_save"]:
        print(f"{timing()} writing regions ({len(regions)})")
        # write_image_to_work("2_1", draw_regions(regions, (H, W)))

    print(f"{timing()} filtering regions by aspect ratio")
    regions = list(filter(lambda r: 0.75 <= r.box.aspect <= 3.0, regions))

    if args["work_save"]:
        print(f"{timing()} writing regions ({len(regions)})")
        # write_image_to_work("2_2", draw_regions(regions, (H, W)))

    print(f"{timing()} removing occluded hole regions")
    regions = remove_occluded_holes(regions, max_boundary_distance=10)

    if args["work_save"]:
        print(f"{timing()} writing regions ({len(regions)})")
        # write_image_to_work("2_3", draw_regions(regions, (H, W)))

    print(f"{timing()} removing highly filled regions")
    regions = list(filter(lambda r: r.fill <= 0.85, regions))

    if args["work_save"]:
        print(f"{timing()} writing regions ({len(regions)})")
        # write_image_to_work("2_4", draw_regions(regions, (H, W)))

    print(f"{timing()} finding chains of similar, adjacent regions")
    chains = find_chains(regions)

    if args["work_save"]:
        print(f"{timing()} writing chains ({len(chains)})")
        img_chains = draw_regions(regions, (H, W))
        for chain in chains:
            chain_box = covering_box([r.box for r in chain])
            cv2.rectangle(
                img_chains, chain_box.tl, chain_box.br, (255, 255, 255), 1)
        # write_image_to_work("3", img_chains)

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
            # write_image_to_work(f"4_{i}", img_roi)

    print(f"{timing()} finding aligned chains")
    aligned_chains = find_aligned_chains(chains)

    if args["work_save"]:
        print(f"{timing()} writing aligned chains")
        chain_boxes = [covering_box([r.box for r in c]) for c in aligned_chains]

        img_ac = img.copy()
        for box in chain_boxes:
            cv2.rectangle(img_ac, box.tl, box.br, (255, 255, 255), 1)

        ac_box = covering_box(chain_boxes)
        cv2.rectangle(img_ac, ac_box.tl, ac_box.br, (255, 255, 255), 2)
        write_image_to_work("5", img_ac)

    print(f"{timing()} finding any missing digits")
    aligned_chains = find_missing_digits(aligned_chains, img_gray)

    if args["work_save"]:
        print(f"{timing()} writing aligned chains (with found digits)")
        chain_boxes = [covering_box([r.box for r in c]) for c in aligned_chains]

        img_ac = img.copy()
        for box in chain_boxes:
            cv2.rectangle(img_ac, box.tl, box.br, (255, 255, 255), 1)

        ac_box = covering_box(chain_boxes)
        cv2.rectangle(img_ac, ac_box.tl, ac_box.br, (255, 255, 255), 2)
        write_image_to_work("6", img_ac)

    print(f"{timing()} finding each chain's associated arrow")
    aligned_chains_arrows = find_arrows(aligned_chains, regions)

    if args["work_save"]:
        print(f"{timing()} writing aligned chains with arrows")
        chain_arrow_boxes = [covering_box([r.box for r in c] + [a.box])
                             for c, a in aligned_chains_arrows]

        img_aca = img.copy()
        for box in chain_arrow_boxes:
            cv2.rectangle(img_aca, box.tl, box.br, (255, 255, 255), 1)

        aca_box = covering_box(chain_arrow_boxes)
        cv2.rectangle(img_aca, aca_box.tl, aca_box.br, (255, 255, 255), 2)
        write_image_to_work("7", img_aca)

        for i, (c, a) in enumerate(aligned_chains_arrows):
            write_image_to_work(f"7_{i}", draw_regions(c + [a]))

    print(f"{timing()} classifying digits and arrows")
    predicted = []
    for chain_digits, arrow in aligned_chains_arrows:

        features_digits = np.array(
            [np.ravel(r.spatial_histogram(3, 5))
             for r in chain_digits])
        predicted_digits = knn_digits.predict(features_digits, k=3)

        features_arrow = np.array([np.ravel(arrow.spatial_histogram(3, 3))])
        predicted_arrow = knn_arrows.predict(features_arrow, k=3)
        predicted.append((predicted_digits, predicted_arrow))

    print(f"{timing()} writing output for {file_root}{file_ext}")
    aca_box = covering_box(
        [covering_box([r.box for r in c] + [a.box])
         for c, a in aligned_chains_arrows])
    img_sign = img[aca_box.indexes]
    cv2.imwrite(f"{args['output']}/DetectedArea{file_id}{file_ext}", img_sign)

    with open(f"{args['output']}/BuildingList{file_id}.txt", "w") as out_file:
        for ds, a in predicted:
            str_digits = "".join(map(str, ds))
            if a[0] == "L":
                str_arrow = "to the left"
            else:
                str_arrow = "to the right"

            print(f"Building {str_digits} {str_arrow}", file=out_file)

    print(f"{timing()} ")
