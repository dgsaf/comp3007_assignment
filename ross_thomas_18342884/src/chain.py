#!/usr/bin/env python

import numpy as np
import cv2

from box import *
from region import *


def linked(region_1, region_2):
    diff_x = np.abs(region_2.box.x - region_1.box.x)
    diff_y = np.abs(region_2.box.y - region_1.box.y)
    diff_height = np.abs(region_2.box.height - region_1.box.height)

    return ((not region_1.box.is_superset_of(region_2.box))
            and diff_y <= 0.5*region_1.box.height
            and diff_y <= 0.5*region_2.box.height
            and diff_height <= 0.2*region_1.box.height
            and diff_height <= 0.2*region_2.box.height
            and diff_x <= 1.0*region_1.box.height
            and diff_x <= 1.0*region_2.box.height)


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
