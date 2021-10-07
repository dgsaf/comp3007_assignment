#!/usr/bin/env python

import numpy as np
import cv2


class Box:
    def __init__(self, box):
        self._x = box[0]
        self._y = box[1]
        self._width = box[2]
        self._height = box[3]

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def tl(self):
        return (self.x, self.y)

    @property
    def br(self):
        return (self.x + self.width, self.y + self.height)

    @property
    def tr(self):
        return (self.x + self.width, self.y)

    @property
    def bl(self):
        return (self.x, self.y + self.height)

    @property
    def center(self):
        return (self.x + int(self.width / 2), self.y + int(self.height / 2))

    @property
    def area(self):
        return (self.width * self.height)

    @property
    def aspect(self):
        return (self.height / self.width)

    @property
    def indexes(self):
        slice_x = slice(self.x, self.x + self.width)
        slice_y = slice(self.y, self.y + self.height)
        return (slice_y, slice_x)

    def contains(self, point):
        x, y = point[0], point[1]
        return (self.x <= x <= self.x + self.width
                and self.y <= y <= self.y + self.height)

    def overlap(self, box):
        a_x = max([self.x, box.x])
        b_x = min([self.x + self.width, box.x + box.width])
        w = b_x - a_x if b_x > a_x else 0

        a_y = max([self.y, box.y])
        b_y = min([self.y + self.height, box.y + box.height])
        h = b_y - a_y if b_y > a_y else 0
        return ((w * h) / box.area)

    def is_superset_of(self, box):
        return (self.x <= box.x <= self.x + self.width - box.width
                and self.y <= box.y <= self.y + self.height - box.height)

    def __str__(self):
        properties = \
            f"Box:\n"\
            + f"tl = ({self.x}, {self.y})\n"\
            + f"width = {self.width}\n"\
            + f"height = {self.height}\n"\
            + f"area = {self.area}\n"\
            + f"aspect = {self.aspect}\n"
        return properties


def covering_box(boxes):
    """
    Construct the smallest box which covers a collection of boxes.

    Parameters
    ----------
    boxes : iterable collection of Box

    Returns
    -------
    cover : Box

    """
    x_min = np.amin([b.x for b in boxes])
    x_max = np.amax([b.x + b.width for b in boxes])
    y_min = np.amin([b.y for b in boxes])
    y_max = np.amax([b.y + b.height for b in boxes])
    cover = Box((x_min, y_min, x_max - x_min, y_max - y_min))
    return cover


def merge_overlapping(boxes, max_overlap=0.05):
    def overlaps(bi, bj):
        return (bi.overlap(bj) >= max_overlap
                or bj.overlap(bi) >= max_overlap)

    def merge_into(boxes, box):
        overlapping = [b for b in boxes if overlaps(box, b)]
        if (len(overlapping) == 0):
            return (boxes + [box])
        else:
            preserved = [b for b in boxes if not overlaps(box, b)]
            merged = covering_box(overlapping + [box])
            return (merge_into(preserved, merged))

    boxes_merged = []
    for b in boxes:
        boxes_merged = merge_into(boxes_merged, b)
    return boxes_merged


def otsu_separation(img_gray, box):
    img_box = (img_gray[box.indexes]).astype(np.uint8)
    t, img_bin = cv2.threshold(img_box, 128, 255, cv2.THRESH_OTSU)

    idxs_w = img_box > t
    idxs_b = img_box <= t

    mean_w = np.average(img_box[idxs_w])
    mean_b = np.average(img_box[idxs_b])
    return (mean_w - mean_b)
