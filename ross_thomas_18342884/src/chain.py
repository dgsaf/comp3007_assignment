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
            rel_diff_height = (rj.box.height - ri.box.height) / ri.box.height
            if ((not ri.box.contains(rj.box))
                and diff_y < 0.75*ri.box.height
                and rel_diff_height < 0.1):
                links[i].append(j)
    return links
