#!/usr/bin/env python

import numpy as np
import cv2


class Box:
    def __init__(self, box):
        self.x = box[0]
        self.y = box[1]
        self.width = box[2]
        self.height = box[3]

    def tl(self):
        return (self.x, self.y)

    def br(self):
        return (self.x + self.width, self.y + self.height)

    def tr(self):
        return (self.x + self.width, self.y)

    def bl(self):
        return (self.x, self.y + self.height)

    def center(self):
        return (self.x + int(self.width / 2), self.y + int(self.height / 2))

    def area(self):
        return (self.width * self.height)

    def aspect(self):
        return (self.height / self.width)

    def contains(self, point):
        x, y = point[0], point[1]
        return (self.x <= x <= self.x + self.width
                and self.y <= y <= self.y + self.height)

    def is_superset_of(self, box):
        return (self.x <= box.x <= self.x + self.width - box.width
                and self.y <= box.y <= self.y + self.height - box.height)

    def __str__(self):
        properties = \
            f"Box:\n"\
            + f"x in [{self.x}, {self.x + self.width}]\n"\
            + f"y in [{self.y}, {self.y + self.height}]\n"
        return properties
