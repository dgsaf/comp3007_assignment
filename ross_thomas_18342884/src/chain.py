#!/usr/bin/env python3

import numpy as np
import cv2

from box import *
from region import *


def linked(region_1, region_2):
    """
    Determine if two regions are sufficiently adjacent and similar.

    Parameters
    ----------
    region_1 : Region
    region_2 : Region

    Returns
    -------
    bool
        True if the bounding boxes of `region_1` and `region_2`
        - have similar `y` placement; and
        - have similar `height`; and
        - have adjacent `x` placement; and
        - do not overlap too much.

    """

    # these are ratios with regard to box heights
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
    """
    Find all non-overlapping chains of regions, in a set of regions.

    Finds all links between regions, then filters the edges by distance between
    to regions to ensure each region has at most one adjacent region to their
    right and at most one adjacent region to their left - ensuring all paths are
    non-overlapping.
    All paths are then extracted and are said to be chains of regions.

    Parameters
    ----------
    regions : list of Region
    best_edge : bool, default=True
        Disabling this results in the edges not being filtered, and so
        overlapping paths may be extracted.

    Returns
    -------
    list of list of Region
        List of chains (represented as a list of regions) of adjacent regions.

    """
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
    """
    Filters a set of chains, to leave the most monochromatic chains.

    For each chain, the minimum Otsu separation of the image across all channels
    (RGB and grayscale), restricted to the bounding box of the chain, is
    calculated.
    The chains are sorted by their separation, and the cluster of most separated
    chains is taken.

    Parameters
    ----------
    img : 3-D array of int
        Colour image.
    chains : list of list of Region
    max_diff : int, default=50
        The cluster of most separated chains is parameterised by this value;
        once ordered by decreasing separation, chains are taken until the change
        in separation from one chain to the next is more than `max_diff`.

    Returns
    -------
    list of list of Region
        The list of chains, ordered by decreasing Otsu separation, which are
        more separated, by `max_diff`, than the remaining list of chains.

    """
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


def aligned(chain_1, chain_2):
    """
    Determine if two chains (of at most 3 regions) are aligned vertically.

    The two chains are considered to be aligned if each of their two right-most
    regions (expected to be digits) are:
    - aligned vertically (or near vertically); and
    - of correspondingly similar heights.

    Parameters
    ----------
    chain_1 : list of Region
    chain_2 : list of Region

    Returns
    -------
    bool
        Flag true if the two chains are aligned.

    """
    n1 = len(chain_1)
    n2 = len(chain_2)

    if (not (2 <= n1 <= 3)) and (not (2 <= n2 <= 3)):
        return False

    norm = lambda p: np.sqrt(np.abs((p[0] ** 2) + (p[1] ** 2)))
    diff = lambda p1, p2: (p1[0] - p2[0], p1[1] - p2[1])

    box_1 = covering_box([r.box for r in chain_1])
    box_2 = covering_box([r.box for r in chain_2])
    w = min([box_1.width, box_2.width])

    n = min([n1, n2])
    digits_1 = {i: chain_1[n1-i-1] for i in range(n)}
    digits_2 = {i: chain_2[n2-i-1] for i in range(n)}

    aligned_vert = np.all(
        [np.abs(diff(digits_1[i].box.center, digits_2[i].box.center)[0])
         <= w
         for i in range(n)])

    similar_heights = np.all(
        [np.abs(digits_1[i].box.height - digits_2[i].box.height)
         <= 0.2 * min([digits_1[i].box.height, digits_2[i].box.height])
         for i in range(n)])

    return (aligned_vert and similar_heights)


def find_aligned_chains(chains):
    """
    Summary

    Parameters
    ----------
    x : t
        e

    Returns
    -------
    f : t
        e

    """
    chains_ordered = sorted(
        chains, key=lambda c: covering_box([r.box for r in c]).y)
    n = len(chains_ordered)

    edges = dict()
    for i, ci in enumerate(chains_ordered[0:n]):
        edges[i] = {j for j, cj in enumerate(chains_ordered)
                    if (j != i and aligned(ci, cj))}

    def dfs(explored, i):
        idxs = {i}
        if i not in explored:
            explored |= {i}
            for j in edges[i]:
                idxs |= dfs(explored, j)
        return idxs

    k = 0
    eq_classes = dict()
    explored = set()
    for i in range(n):
        if i not in explored:
            eq_classes[k] = dfs(explored, i)
            k += 1

    aligned_chains = [[chains_ordered[j]
                       for j in eq_classes[k]]
                      for k in eq_classes
                      if len(eq_classes[k]) > 1]

    return max(aligned_chains, key=lambda ac: len(ac))


def find_missing_digits(aligned_chains, img_gray):
    """
    Summary

    Parameters
    ----------
    x : t
        e

    Returns
    -------
    f : t
        e

    """
    H, W = img_gray.shape[:2]

    aligned_chains_found = []
    for chain in aligned_chains:
        n = len(chain)

        if n == 3:
            aligned_chains_found.append(chain)
            continue

        digit_2 = chain[n-2]
        digit_3 = chain[n-1]

        box = covering_box([digit_2.box, digit_3.box])
        diff_x = digit_3.box.tl[0] - digit_2.box.tl[0]
        diff_y = digit_3.box.tl[1] - digit_2.box.tl[1]

        x = max([0, box.tl[0] - int(0.9*diff_x)])
        y = min([H - 1, max([0, box.tl[1] - diff_y])])
        w = box.tl[0] - x - 1
        h = min([H - y - 1, max([digit_2.box.height, digit_3.box.height])])
        left_box = Box(x, y, w, h)

        img_box = (img_gray[left_box.indexes]).astype(np.uint8)
        t, img_bin = cv2.threshold(img_box, 128, 255, cv2.THRESH_OTSU)
        regions = cc_regions(img_bin)

        region = max(regions, key=lambda r: r.area)
        digit_1 = Region({(x + p[0], y + p[1]) for p in region.points})

        aligned_chains_found.append([digit_1, digit_2, digit_3])
    return aligned_chains_found


def find_arrows(aligned_chains, regions):
    """
    Summary

    Parameters
    ----------
    x : t
        e

    Returns
    -------
    f : t
        e

    """
    aligned_chains_arrows = []
    for chain in aligned_chains:
        n = len(chain)

        digit_2 = chain[n-2]
        digit_3 = chain[n-1]

        box = covering_box([digit_2.box, digit_3.box])
        right_box = Box(box.tr[0] + 1, box.tr[1], box.width, box.height)

        arrow = max(regions, key=lambda r: right_box.overlap(r.box))
        aligned_chains_arrows.append((chain, arrow))
    return aligned_chains_arrows
