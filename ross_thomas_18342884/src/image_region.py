#!/usr/bin/env python

import os
import numpy as np
import cv2


def partition(points):
    neighborhood = np.array(
        [[0, 1], [1, 1], [1, 0], [1, -1],
         [0, -1], [-1, -1], [-1, 0], [-1, 1]])
    boundary = []
    internal = []
    for point in points:
        adjacent = map(lambda n: point + n, neighborhood)
        if np.all(np.isin(adjacent, points)):
            internal.append(point)
        else:
            boundary.append(point)
    return (np.array(internal), np.array(boundary))


class Region:
    def __init__(self, points, box):
        self.points = np.array(points)
        self.box = box

        self.tl = np.array(box[0:1])
        self.width = box[2]
        self.height = box[3]
        self.br = self.tl + np.array([self.width, self.height])
        self.box_area = self.width * self.height

        self.area = self.points.size
        self.fill = self.area / (self.box_area + 1.0e-8)
        self.aspect = self.height / (self.width + 1.0e-8)

        self.internal, self.boundary = partition(self.points)
        self.diameter = np.amax(
            [cv2.norm(bp_1 - bp_2)
             for bp_1 in self.boundary
             for bp_2 in self.boundary])

        def distance(self, point):
            return np.amin([cv2.norm(bp - point) for bp in self.boundary])

        def hausdorff_distance(self, region):
            return np.amin([self.distance(bp) for bp in region.boundary])

        def image(self):
            img = np.zeros((self.width, self.height), dtype=np.uint8)
            for point in self.points:
                img[point - self.tl] = 255
            return img

        def contours(self):
            _, contours, hierarchy = cv2.findContours(
                self.image(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            return (contours, hierarchy)

        self.holes = len((self.contours())[0]) - 1
        self.moments = cv2.moments(self.image(), binaryImage=True)
        self.hu_moments = cv2.HuMoments(self.moments)
