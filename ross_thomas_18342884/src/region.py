#!/usr/bin/env python

import numpy as np
import cv2
import random

from box import *


class Region:
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
        return (m['m10'] / m['m00'], m['m01'] / m['m00'])

    @property
    def hu_moments(self):
        return cv2.HuMoments(self.moments)[:, 0]

    @property
    def hu_moments_regular(self):
        return np.array([- np.sign(h) * np.log(np.abs(h))
                         for h in self.hu_moments])

    @property
    def covariance(self):
        m = self.moments
        cov = np.zeros((2, 2), dtype=np.float32)
        cov[0, 0] = m["mu02"] / m["mu00"]
        cov[0, 1] = m["mu11"] / m["mu00"]
        cov[1, 0] = m["mu11"] / m["mu00"]
        cov[1, 1] = m["mu20"] / m["mu00"]
        return cov


    @property
    def covariance_eigen(self):
        cov = self.covariance
        a = (cov[0, 0] + cov[1, 1]) / 2
        b = (cov[0, 0] - cov[1, 1]) / 2
        c = np.sqrt(np.abs((cov[0, 1] ** 2) + (b ** 2)))
        eig_1, eig_2 = a + c, a - c
        theta = np.arctan(cov[0, 1] / b) / 2
        return (eig_1, eig_2, theta)

    @property
    def features(self):
        eig_1, eig_2, theta = self.covariance_eigen
        return np.append(self.hu_moments, [self.holes, eig_1, eig_2, theta])

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
            + f"holes = {self.holes}\n"\
            + f"hu moments (reg) = \n{self.hu_moments_regular}\n"\
            + f"covariance eig = {self.covariance_eigen}"
        return properties


def remove_overlapping(regions, max_overlap=0.8):
    regions_ordered = sorted(regions, key=lambda r: r.area, reverse=True)
    regions_filtered = []
    for r in regions_ordered:
        if np.all([not (rf.box.is_superset_of(r.box)
                        and rf.overlap(r) >= max_overlap)
                   for rf in regions_filtered]):
            regions_filtered.append(r)
    return regions_filtered


def remove_occluded_holes(regions, max_boundary_distance=10):
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
    n, labels = cv2.connectedComponents(img_bin, connectivity=8)
    regions = [Region(np.argwhere(np.transpose(labels) == l))
               for l in range(1, n)]
    return regions
