#!/usr/bin/env python3

import numpy as np
import cv2


class Box:
    """
    Rectangular box, suitable for use with OpenCV.

    Attributes
    ----------
    x : int
        x-coordinate of top-left corner
    y : int
        y-coordinate of top-left corner
    width : int
        x-length of box
    height : int
        y-length of box

    Methods
    -------
    tl : (int, int)
        Top left corner of the box.

    br : (int, int)
        Bottom right corner of the box.

    tr : (int, int)
        Top right corner of the box.

    bl : (int, int)
        Bottom left corner of the box.

    center : (int, int)
        Geometric center point of the box.

    area : int
        Rectangular area of the box.

    aspect : float
        Aspect ratio of the box, in height/width format.

    indexes : (slice int, slice int)
        2-D array index slices that this box corresponds to.
        Suitable for accessing 2-D arrays as `array[box.indexes]`.

    contains(point) : bool
        Returns true if `point` is within the box.

    overlap(box) : float
        The fractional of the area of `box` that overlaps with this box.

    is_superset_of(box) : bool
        Returns true if `box` is entirely inside this box.

    Notes
    -----
    As per OpenCV, a point `(x, y)` corresponds to an array index `array[y, x]`.

    """

    def __init__(self, x, y, width, height):
        self._x = x
        self._y = y
        self._width = width
        self._height = height

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
            + f"aspect = {self.aspect}"
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
    cover = Box(x_min, y_min, x_max - x_min, y_max - y_min)
    return cover


def bounding_box(points):
    """
    Construct the minimal bounding box for a given set of 2-D points.

    Parameters
    ----------
    points : iterable collection of (int, int)

    Returns
    -------
    bounding : Box

    """
    x, y, w, h = cv2.boundingRect(np.array([p for p in points]))
    bounding = Box(x, y, w, h)
    return bounding


def merge_overlapping(boxes, max_overlap=0.05):
    """
    Merge all sufficiently overlapping boxes in a collection of boxes.

    Parameters
    ----------
    boxes : iterable collection of Box
    max_overlap : float, default=0.05
        Merge any pair of boxes for which one of them overlaps the other more
        than this value.

    Returns
    -------
    boxes_merged : list of Box

    """
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
    """
    Calculate the Otsu separation of a single-channel image restricted to a box.

    Parameters
    ----------
    img_gray : 2-D array
        Single channel image.
    box : Box
        The 2-D restriction of `img_gray` for which the Otsu separation is
        calculated.

    Returns
    -------
    otsu_sep : float
        The separation between the black and white class means, after
        determining class by Otsu thresholding.

    """
    img_box = (img_gray[box.indexes]).astype(np.uint8)
    t, img_bin = cv2.threshold(img_box, 128, 255, cv2.THRESH_OTSU)

    idxs_w = img_box > t
    idxs_b = img_box <= t

    mean_w = np.average(img_box[idxs_w])
    mean_b = np.average(img_box[idxs_b])
    otsu_sep = mean_w - mean_b
    return otsu_sep


def otsu_separation_color(img, box):
    """
    Calculate the Otsu separation of a colour image restricted to a box.

    Parameters
    ----------
    img : 3-D array
        Color image with 3 colour channels.
    box : Box
        The 2-D restriction of `img` for which the Otsu separation is
        calculated.

    Returns
    -------
    min_otsu_sep : float
        The minimum of the Otsu separations calculated for each colour channel
        of `img` and its grayscale transformation.

    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    min_otsu_sep = np.amin(
        [otsu_separation(img_gray, box),
         otsu_separation(img[:, :, 0], box),
         otsu_separation(img[:, :, 1], box),
         otsu_separation(img[:, :, 2], box)])
    return min_otsu_sep
