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

    def area(self):
        return (self.width * self.height)

    def aspect(self):
        return (self.height / (self.width + 1.0e-8))
