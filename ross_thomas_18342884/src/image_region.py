#!/usr/bin/env python

import os
import numpy as np
import cv2


class Region:
    def __init__(self, points, box):
        self.points = np.array(points)
        self.box = box

        self.tl = np.array(box[0:2])
        self.width = box[2]
        self.height = box[3]
        self.br = self.tl + np.array([self.width, self.height])
        self.box_area = self.width * self.height

        self.area = self.points.size
        self.fill = self.area / (self.box_area + 1.0e-8)
        self.aspect = self.height / (self.width + 1.0e-8)

        self.boundary = np.concatenate(
            [np.reshape(contour, (-1, 2))
             for contour in self.contours()[0]])

        self.diameter = np.amax(
            [cv2.norm(bp_1 - bp_2)
             for bp_1 in self.boundary
             for bp_2 in self.boundary])

        self.holes = len((self.contours())[0]) - 1
        self.moments = cv2.moments(self.image(), binaryImage=True)
        self.hu_moments = cv2.HuMoments(self.moments)

    def distance(self, point):
        return np.amin([cv2.norm(bp - point) for bp in self.boundary])

    def hausdorff_distance(self, region):
        return np.amin([self.distance(bp) for bp in region.boundary])

    def image(self):
        img = np.zeros((self.height, self.width), dtype=np.uint8)
        for point in self.points:
            j, i = tuple(point - self.tl)
            img[i, j] = 255
        return img

    def contours(self):
        _, contours, hierarchy = cv2.findContours(
            self.image(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return (contours, hierarchy)

    def display(self):
        details = \
            f"Region:\n"\
            + f"  tl = ({self.tl[0]} , {self.tl[1]})\n"\
            + f"  size = ({self.width} x {self.height})\n"\
            + f"  area = {self.area}\n"\
            + f"  fill = {self.fill}\n"\
            + f"  aspect = {self.aspect}\n"\
            + f"  diameter = {self.diameter}\n"\
            + f"  holes = {self.holes}\n"\
            + f"  moments = {self.moments}\n"\
            + f"  hu moments = {self.hu_moments}\n"

        print(details)
        cv2.imshow("region", self.image())
        cv2.waitKey(0)
        cv2.destroyWindow("region")
