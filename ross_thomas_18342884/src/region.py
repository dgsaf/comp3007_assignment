#!/usr/bin/env python

import os
import numpy as np
import cv2


class Region:
    def __init__(self, points, box):
        self.points = set([(p[0], p[1]) for p in points])
        self.box = box

        self.x = box[0]
        self.y = box[1]
        self.width = box[2]
        self.height = box[3]
        self.tl = (self.x, self.y)
        self.br = (self.x + self.width, self.y + self.height)
        self.box_area = self.width * self.height

        self.area = len(self.points)
        self.fill = self.area / (self.box_area + 1.0e-8)
        self.aspect = self.height / (self.width + 1.0e-8)

        self.boundary = set(
            [(p[0], p[1])
             for p in
             (np.concatenate(
                 [np.reshape(c, (-1, 2)) for c in self.contours()[0]]))])

        self.holes = len((self.contours())[0]) - 1
        self.moments = cv2.moments(self.image(), binaryImage=True)
        self.hu_moments = cv2.HuMoments(self.moments)
        self.hu_moments_regular = map(
            lambda h: -np.sign(h) * np.log(np.abs(h)), self.hu_moments)

    def image(self):
        img = np.zeros((self.height, self.width), dtype=np.uint8)
        for point in self.points:
            j, i = (point[0] - self.x, point[1] - self.y)
            img[i, j] = 255
        return img

    def contours(self):
        _, contours, hierarchy = cv2.findContours(
            self.image(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return (contours, hierarchy)

    def distance(self, point):
        return np.amin([cv2.norm(bp - point) for bp in self.boundary])

    def hausdorff_distance(self, region):
        return np.amin([self.distance(bp) for bp in region.boundary])

    def overlap(self, region):
        return len(self.points.intersection(region.points)) / len(region.points)

    def contains(self, region):
        return np.all([(bp in self.points) for bp in region.boundary])

    def display(self):
        details = \
            f"Region:\n"\
            + f"  x, y = ({self.x} , {self.y})\n"\
            + f"  size = ({self.width} x {self.height})\n"\
            + f"  area = {self.area}\n"\
            + f"  fill = {self.fill}\n"\
            + f"  aspect = {self.aspect}\n"\
            + f"  holes = {self.holes}\n"\
            + f"  moments = {self.moments}\n"\
            + f"  hu moments = {self.hu_moments}\n"\
            + f"  (regular) hu moments = {self.hu_moments_regular}\n"

        print(details)
        cv2.imshow("region", self.image())
        cv2.waitKey(0)
        cv2.destroyWindow("region")


def unique_regions(regions, threshold=0.8):
    regions_sorted = sorted(regions, key=lambda r: r.area, reverse=True)
    regions_unique = []
    for r in regions_sorted:
        if all([ur.overlap(r) < threshold for ur in regions_unique]):
            regions_unique.append(r)
    return regions_unique


def mser_regions(img_gray, min_area=25, max_area=2000, delta=20, threshold=0.8):
    mser = cv2.MSER_create()
    mser.setMinArea(min_area)
    mser.setMaxArea(max_area)
    mser.setDelta(delta)

    point_sets, boxes = mser.detectRegions(img_gray)
    regions = [Region(ps, b) for (ps, b) in zip(point_sets, boxes)]
    regions_unique = unique_regions(regions, threshold=threshold)
    return regions_unique
