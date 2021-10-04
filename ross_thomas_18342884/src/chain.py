#!/usr/bin/env python

import numpy as np
import cv2

from box import *
from region import *


# find chains of regions
# - left to right
# - not too far apart vertically
# - similar height
def find_chains(regions):
    regions_ordered = sorted(regions, key=lambda r: r.box.x)
    n = len(regions_ordered)

    links = [[] for i in range(n)]
    for i in range(n):
        ri = regions_ordered[i]
        for j in range(i+1, n):
            rj = regions_ordered[j]

            diff_y = np.abs(rj.box.y - ri.box.y)
            diff_height = np.abs(rj.box.height - ri.box.height)
            diff_x = np.abs(rj.box.x - ri.box.x)

            if ((not ri.box.contains(rj.box))
                and diff_y <= 0.5*ri.box.height
                and diff_y <= 0.5*rj.box.height
                and diff_height < 0.05*ri.box.height
                and diff_height < 0.05*rj.box.height
                and diff_x <= 1.5*ri.box.height):
                links[i].append(j)
    return links
