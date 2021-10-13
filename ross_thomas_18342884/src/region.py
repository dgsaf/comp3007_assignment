#!/usr/bin/env python3

import numpy as np
import cv2
import math
import random

from box import *


class Region:
    """
    Connected region of points, suitable for use with OpenCV MSER.

    Attributes
    ----------
    points : set of (int, int)
        Set of points that this region contains.
        These points are assumed, but not checked, upon construction to be
        connected.

    box : Box
        Minimal bounding box of this region.

    boundary : set of (int, int)
        Set of points of this region which are adjacent to at least point not in
        this region.
        Calculated using OpenCV's `findContours()` method.
        Is calculated as needed, and then cached.

    cached_boundary : bool
        Flag true if the boundary points have been calculated.

    contours : list of list of (int, int)
        The set of contours calculated using OpenCV's `findContours()` method.
        Is calculated as needed, and then cached.

    hierarchy : 2-D array
        The heirarchy of the set of contours calculated using OpenCV's
        `findContours()` method.
        Is calculated as needed, and then cached.

    cached_contours : bool
        Flag true if the contours have been calculated.

    Methods
    -------
    area : int
        The cardinality of `points`; that is, the number of points this region
        contains.

    fill : float
        The fraction of the area of the bounding box of this region that this
        region also contains.

    holes : int
        The number of interior holes this regions contains.
        Inferred from the number of contours this region has.

    moments : dict of (string, float)
        The image moments of this region, assuming each point has a mass of 1.
        Calculated using OpenCV's `moments()` method.

    centroid : (float, float)
        The mass centre of this region, calculated from its moments.

    hu_moments : array of float
        The Hu moments of this region, calculated using OpenCV's `HuMoments()`
        method.

    image : 2-D array of int
        Represents this region as binary image, with dimensions of its bounding
        box, with a point in the image being white if it is in `points`.

    spatial_occupancy(bins_x, bins_y) : 2-D array of float
        Constructs a set of bins, symmetric about the geometric centre of `box`,
        with `bins_x` x-bins and `bins_y` y-bins, then calculates the fill of
        each bin by the region.

    distance(point) : float
        Calculates the minimum distance of `point` to any of the points in
        this region.

    set_distance_min(points) : float
        Calculates the minimum of all distances between `points` and points in
        this region.

    set_distance_max(points) : float
        Calculates the maximum of all distances between `points` and points in
        this region.

    overlap(region) : float
        Calculates the fractional cardinality of the intersection of `region`
        with this region, over the cardinality of `region`.

    contains(region) : bool
        Returns true if all points in `region` are also in this region.

    """

    def __init__(self, points):
        self._points = set([(p[0], p[1]) for p in points])
        self._box = bounding_box(self.points)

        self._cached_boundary = False
        self._boundary = None

        self._cached_contours = False
        self._contours = None
        self._hierarchy = None

    @property
    def points(self):
        return self._points

    @property
    def box(self):
        return self._box

    @property
    def area(self):
        return len(self.points)

    @property
    def fill(self):
        return (self.area / self.box.area)

    @property
    def boundary(self):
        if not self._cached_boundary:
            cs = [np.reshape(c, (-1, 2)) for c in (self.contours[0])]
            self._boundary = set(
                [(self.box.x + p[0], self.box.y + p[1])
                 for p in np.concatenate(cs)])
            self._cached_boundary = True
        return self._boundary

    @property
    def holes(self):
        return (len(self.contours[0]) - 1)

    @property
    def moments(self):
        return cv2.moments(self.image().astype(np.float32), binaryImage=True)

    @property
    def centroid(self):
        m = self.moments
        return (m["m10"] / m["m00"], m["m01"] / m["m00"])

    @property
    def hu_moments(self):
        return cv2.HuMoments(self.moments)[:, 0]

    @property
    def contours(self):
        if not self._cached_contours:
            _, self._contours, self._hierarchy = cv2.findContours(
                self.image().astype(np.uint8),
                cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            self._cached_contours = True
        return (self._contours, self._hierarchy)

    def image(self):
        img = np.zeros((self.box.height, self.box.width), dtype=np.uint8)
        for point in self.points:
            j, i = (point[0] - self.box.x, point[1] - self.box.y)
            img[i, j] = 255
        return img

    def spatial_occupancy(self, bins_x, bins_y):
        s_x = math.ceil(self.box.width / bins_x)
        s_y = math.ceil(self.box.height / bins_y)

        c_x = math.floor((self.box.width - ((bins_x - 2) * s_x)) / 2)
        c_y = math.floor((self.box.height - ((bins_y - 2) * s_y)) / 2)

        xs = np.array(
            [0]
            + [(i * s_x) + c_x for i in range(0, bins_x - 1)]
            + [self.box.width])
        ys = np.array(
            [0]
            + [(i * s_y) + c_y for i in range(0, bins_y - 1)]
            + [self.box.height])

        img = self.image()
        bins = np.zeros((bins_y, bins_x), dtype=np.float32)
        for i in range(bins_y):
            sl_y = slice(ys[i], ys[i+1])
            for j in range(bins_x):
                sl_x = slice(xs[j], xs[j+1])
                n = (ys[i+1] - ys[i]) * (xs[j+1] - xs[j])
                bins[i, j] = np.count_nonzero(img[sl_y, sl_x]) / n
        return bins

    def distance(self, point):
        if point in self.points:
            min_distance = 0.0
        else:
            norm = lambda p: np.sqrt(np.abs((p[0] ** 2) + (p[1] ** 2)))
            diff = lambda p1, p2: (p1[0] - p2[0], p1[1] - p2[1])

            min_distance = np.amin(
                [norm(diff(bp, point)) for bp in self.boundary])
        return min_distance

    def set_distance_min(self, points):
        return np.amin([self.distance(p) for p in points])

    def set_distance_max(self, points):
        return np.amax([self.distance(p) for p in points])

    def overlap(self, region):
        return len(self.points.intersection(region.points)) / len(region.points)

    def contains(self, region):
        return np.all([(p in self.points) for p in region.points])

    def show(self):
        cv2.imshow("region", self.image())
        cv2.waitKey(0)
        cv2.destroyWindow("region")
        return

    def __str__(self):
        properties = \
            f"Region:\n"\
            + f"{str(self.box)}\n"\
            + f"area = {self.area}\n"\
            + f"fill = {self.fill}\n"\
            + f"holes = {self.holes}"
        return properties


def remove_overlapping(regions, max_overlap=0.8):
    """
    Filters regions by removing sufficiently overlapping smaller regions.

    Parameters
    ----------
    regions : iterable collection of Region
    max_overlap : float, default=0.8
        Remove any region which overlaps with a larger region, by area, by more
        than this value.

    Returns
    -------
    regions_filtered : iterable collection of Region

    """
    regions_ordered = sorted(regions, key=lambda r: r.area, reverse=True)
    regions_filtered = []
    for r in regions_ordered:
        if np.all([not (rf.box.is_superset_of(r.box)
                        and rf.overlap(r) >= max_overlap)
                   for rf in regions_filtered]):
            regions_filtered.append(r)
    return regions_filtered


def remove_occluded_holes(regions, max_boundary_distance=10):
    """
    Filters interior hole regions, which fill up another regions hole.

    Parameters
    ----------
    regions : iterable collection of Region
    max_boundary_distance : int, default=10
        Remove any region with boundary points which are never more than this
        distance away from another region which contains this one.

    Returns
    -------
    regions_filtered : iterable collection of Region

    """
    regions_ordered = sorted(regions, key=lambda r: r.box.x)
    regions_filtered = []
    for r in regions_ordered:
        occludes = lambda rf: np.all(
            [rf.distance(bp) <= max_boundary_distance for bp in r.boundary])
        if np.all([not (rf.box.is_superset_of(r.box) and occludes(rf))
                   for rf in regions_filtered]):
            regions_filtered.append(r)
    return regions_filtered


def draw_regions(regions, size=None):
    """
    Creates an image from a set of regions.

    Parameters
    ----------
    regions : iterable collection of Region
    size : (int, int), optional
        Height and width of the image canvas, on which to draw the regions.

    Returns
    -------
    img_regions : 3-D array of int
        Colour image (constructed in HSV space, but returned in BGR) with a
        black background, distinct colours for each region, and with the
        boundaries of regions coloured white.

    """
    if size:
        canvas = Box(0, 0, size[1], size[0])
    else:
        canvas = covering_box([r.box for r in regions])

    img_regions = np.zeros(
        (canvas.height, canvas.width, 3), dtype=np.uint8)
    for r in regions:
        color = (random.randint(0, 179), 255, 255)
        for p in r.points:
            x, y = p
            img_regions[y - canvas.y, x - canvas.x] = color
        for bp in r.boundary:
            x, y = bp
            img_regions[y - canvas.y, x - canvas.x] = (0, 0, 255)
    img_regions = cv2.cvtColor(img_regions, cv2.COLOR_HSV2BGR)
    return img_regions


def cc_regions(img_bin):
    """
    Creates a set of regions from the connected components of a binary image.

    Parameters
    ----------
    img_bin : 2-D array of int
        Binary image.

    Returns
    -------
    regions : list of Region
        The set of regions formed from connected components of a binary image.

    """
    n, labels = cv2.connectedComponents(img_bin, connectivity=8)
    regions = [Region(np.argwhere(np.transpose(labels) == l))
               for l in range(1, n)]
    return regions
