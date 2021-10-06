#!/usr/bin/env python

import numpy as np
import cv2
import random

from box import *


class Region:
    def __init__(self, points, box):
        self.points = set([(p[0], p[1]) for p in points])
        self.box = Box(box)

        self.cached_boundary = False
        self._boundary = set()
        # self.boundary = set(
        #     [(self.box.x + p[0], self.box.y + p[1])
        #      for p in
        #      (np.concatenate(
        #          [np.reshape(c, (-1, 2)) for c in self.contours()[0]]))])

    def area(self):
        return len(self.points)

    def fill(self):
        return (self.area() / self.box.area())

    def image(self):
        img = np.zeros((self.box.height, self.box.width), dtype=np.uint8)
        for point in self.points:
            j, i = (point[0] - self.box.x, point[1] - self.box.y)
            img[i, j] = 255
        return img

    def contours(self):
        _, contours, hierarchy = cv2.findContours(
            self.image().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return (contours, hierarchy)

    def boundary(self):
        if not self.cached_boundary:
            self._boundary = set(
                [(self.box.x + p[0], self.box.y + p[1])
                 for p in
                 (np.concatenate(
                     [np.reshape(c, (-1, 2)) for c in self.contours()[0]]))])
            self.cached_boundary = True
        return self._boundary

    def holes(self):
        return (len((self.contours())[0]) - 1)

    def moments(self):
        return cv2.moments(self.image().astype(np.float32), binaryImage=True)

    def hu_moments(self):
        return cv2.HuMoments(self.moments())

    def hu_moments_regular(self):
        h = self.hu_moments()
        return (-np.sign(h) * np.log(np.abs(h)))

    def distance(self, point):
        if point in self.points:
            min_distance = 0.0
        else:
            norm = lambda p: np.sqrt(np.abs((p[0] ** 2) + (p[1] ** 2)))
            diff = lambda p1, p2: (p1[0] - p2[0], p1[1] - p2[1])

            min_distance = np.amin(
                [norm(diff(bp, point)) for bp in self.boundary()])
        return min_distance

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
            + f"{str(self.box)}"\
            + f"area = {self.area()}\n"\
            + f"fill = {self.fill()}\n"\
            + f"holes = {self.holes()}\n"\
            + f"hu moments = \n{self.hu_moments()}\n"
        return properties


def remove_overlapping(regions, max_overlap=0.8):
    regions_ordered = sorted(regions, key=lambda r: r.area(), reverse=True)
    regions_filtered = []
    for r in regions_ordered:
        if all([ru.overlap(r) < max_overlap for ru in regions_filtered]):
            regions_filtered.append(r)
    return regions_filtered


def remove_occluded_holes(regions, max_boundary_distance=10):
    regions_ordered = sorted(regions, key=lambda r: r.box.x)
    regions_filtered = []
    for r in regions_ordered:
        occludes = lambda rf: np.all(
            [rf.distance(bp) <= max_boundary_distance for bp in r.boundary()])
        if np.all([not (rf.box.is_superset_of(r.box) and occludes(rf))
                   for rf in regions_filtered]):
            regions_filtered.append(r)
    return regions_filtered


def draw_regions(regions, size=None):
    if size:
        canvas = Box((0, 0, size[1], size[0]))
    else:
        canvas = covering_box([r.box for r in regions])

    img_regions = np.zeros(
        (canvas.height, canvas.width, 3), dtype=np.uint8)
    for r in regions:
        color = (random.randint(0, 179), 255, 255)
        for p in r.points:
            x, y = p
            img_regions[y - canvas.y, x - canvas.x] = color
        for bp in r.boundary():
            x, y = bp
            img_regions[y - canvas.y, x - canvas.x] = (0, 0, 255)
    img_regions = cv2.cvtColor(img_regions, cv2.COLOR_HSV2BGR)
    return img_regions
