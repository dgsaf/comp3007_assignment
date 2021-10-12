#!/usr/bin/env python3

import numpy as np
import cv2

from box import *
from region import *


def linked(region_1, region_2):
    # these are ratios w.r.t. box heights
    max_ratio_diff_y = 0.5
    max_ratio_diff_height = 0.2
    max_ratio_diff_x = 1.0
    max_overlap = 0.25

    diff_x = np.abs(region_2.box.x - region_1.box.x)
    diff_y = np.abs(region_2.box.y - region_1.box.y)
    diff_height = np.abs(region_2.box.height - region_1.box.height)

    similar_height = (
        diff_height <= max_ratio_diff_height*region_1.box.height
        and diff_height <= max_ratio_diff_height*region_2.box.height)

    similar_y = (
        diff_y <= max_ratio_diff_y*region_1.box.height
        and diff_y <= max_ratio_diff_y*region_2.box.height)

    adjacent_x = (
        diff_x <= max_ratio_diff_x*region_1.box.height
        and diff_x <= max_ratio_diff_x*region_2.box.height)

    non_occluding = (
        (not region_1.box.is_superset_of(region_2.box))
        and region_1.box.overlap(region_2.box) <= max_overlap
        and region_2.box.overlap(region_1.box) <= max_overlap)

    return (similar_height and similar_y and adjacent_x and non_occluding)


def find_chains(regions, best_edge=True):
    regions_ordered = sorted(regions, key=lambda r: r.box.x)
    n = len(regions_ordered)

    edges = dict()
    roots = set(range(0, n))
    for i in range(n):
        ri = regions_ordered[i]

        links = {j for j in range(i+1, n) if linked(ri, regions_ordered[j])}
        if best_edge:
            links_sorted = sorted(
                links,
                key=(lambda j: ri.distance(regions_ordered[j].box.center)))
            links = set(links_sorted[:1])

        edges[i] = links
        roots -= edges[i]
        if not edges[i]:
            roots -= {i}

    def paths(i):
        if edges[i]:
            suffixes = [p for j in edges[i] for p in paths(j)]
            return [[i] + s for s in suffixes]
        else:
            return [[i]]

    chains = [p for i in roots for p in paths(i)]
    return [list(map(lambda i: regions_ordered[i], c)) for c in chains]


def cluster_largest_otsu_separations(img, chains, max_diff=50):
    if not chains:
        return chains

    chains_ordered = sorted(
        chains,
        key=(lambda c: \
             otsu_separation_color(img, covering_box([r.box for r in c]))),
        reverse=True)

    otsu_seps = np.array(
        [otsu_separation_color(img, covering_box([r.box for r in c]))
         for c in chains_ordered])

    n = len(otsu_seps)
    idx = n-1
    for i in range(n-1):
        diff = np.abs(otsu_seps[i+1] - otsu_seps[i])
        if diff >= max_diff:
            idx = i
            break

    return (chains_ordered[:idx+1])
