#!/usr/bin/env python

import numpy as np
import cv2

from box import *


class Region:
    def __init__(self, points, box):
        self.points = set([(p[0], p[1]) for p in points])
        self.box = Box(box)

        self.boundary = set(
            [(p[0], p[1])
             for p in
             (np.concatenate(
                 [np.reshape(c, (-1, 2)) for c in self.contours()[0]]))])

    def area(self):
        return len(self.points)

    def fill(self):
        return (self.area() / self.box.area())

    def image(self):
        img = np.zeros((self.box.height, self.box.width), dtype=np.uint8)
        for point in self.points:
            j, i = (point[0] - self.box.x, point[1] - self.box.y)
            img[i, j] = 255.0
        return img

    def contours(self):
        _, contours, hierarchy = cv2.findContours(
            self.image().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return (contours, hierarchy)

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
        return np.amin([cv2.norm(bp - point) for bp in self.boundary])

    def hausdorff_distance(self, region):
        return np.amin([self.distance(bp) for bp in region.boundary])

    def overlap(self, region):
        return len(self.points.intersection(region.points)) / len(region.points)

    def contains(self, region):
        return np.all([(bp in self.points) for bp in region.boundary])

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


def unique_regions(regions, threshold=0.8):
    regions_sorted = sorted(regions, key=lambda r: r.area(), reverse=True)
    regions_unique = []
    for r in regions_sorted:
        if all([ur.overlap(r) < threshold for ur in regions_unique]):
            regions_unique.append(r)
    return regions_unique


def mser_regions(img_gray, min_area=25, max_area=2000, delta=5, threshold=0.8):
    mser = cv2.MSER_create()
    mser.setMinArea(min_area)
    mser.setMaxArea(max_area)
    mser.setDelta(delta)

    point_sets, boxes = mser.detectRegions(img_gray)
    regions = [Region(ps, b) for (ps, b) in zip(point_sets, boxes)]
    regions_unique = unique_regions(regions, threshold=threshold)
    return regions_unique
