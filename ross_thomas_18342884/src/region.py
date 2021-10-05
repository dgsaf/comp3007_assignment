#!/usr/bin/env python

import numpy as np
import cv2
import random

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
        if point in self.points:
            min_distance = 0.0
        else:
            min_distance = np.amin(
                [cv2.norm(bp - point) for bp in self.boundary])
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


def remove_overlapping(regions, threshold=0.8):
    regions_sorted = sorted(regions, key=lambda r: r.area(), reverse=True)
    regions_unique = []
    for r in regions_sorted:
        if all([ur.overlap(r) < threshold for ur in regions_unique]):
            regions_unique.append(r)
    return regions_unique


def draw_regions(regions, size=None):
    if size:
        canvas = Box((0, 0, size[1], size[0]))
    else:
        canvas = covering_box([r.box for r in regions])

    img_regions = np.zeros(
        (canvas.height, canvas.width, 3), dtype=np.uint8)
    for r in regions:
        color = (random.randint(100, 255),
                 random.randint(100, 255),
                 random.randint(100, 255))
        for p in r.points:
            x, y = point
            img_regions[y - canvas.y, x - canvas.x] = color
        for bp in r.boundary:
            x, y = bp
            img_regions[y - canvas.y, x - canvas.x] = (255, 255, 255)
    return img_regions
