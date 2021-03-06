#!/usr/bin/env python3

import os
import numpy as np
import cv2

from region import *


class KNN:
    """
    k-Nearest Neighbour class, built around OpenCV's KNearest object.

    Attributes
    ----------
    labels : dict of (int, X), where X is type of label of samples
        A map between the int labels used internally, and the labels of the
        samples as provided to the `train()` method.
    knn : cv2.ml.KNearest
        The k-Nearest Neighbour object, trained on the sample data, which is
        used to predict class labels for unlabelled sample data.

    Methods
    -------
    train(samples_labelled) :
        Train `knn` on the labelled sample data, and build the map `labels`,
        between internal labels for OpenCV's `train()` and `predict()` methods,
        and the labels of `samples_labelled`.

    predict(samples, k=3) :
        Predict the class labels of `samples` using `k` neighbours, and the
        trained `knn` object.
        The output labels are converted from the internal class labels to be of
        the same type as was provided during training.

    """

    def __init__(self, samples_labelled):
        self.train(samples_labelled)
        return

    @property
    def labels(self):
        return self._labels

    @property
    def knn(self):
        return self._knn

    def train(self, samples_labelled):
        self._labels = {k : l for k, l in enumerate(samples_labelled.keys())}

        samples = np.concatenate(
            [samples_labelled[self.labels[k]].astype(np.float32)
             for k in self.labels])

        responses = np.concatenate(
            [np.full((samples_labelled[self.labels[k]].shape[0]), k)
             for k in self.labels])

        self._knn = cv2.ml.KNearest_create()
        self._knn.setIsClassifier(True)
        self._knn.setAlgorithmType(cv2.ml.KNearest_BRUTE_FORCE)
        self._knn.train(samples, cv2.ml.ROW_SAMPLE, responses)
        return

    def predict(self, samples, k=3):
        _, responses, _, dist = self.knn.findNearest(samples, k)

        labels_predicted = np.array(
            [self.labels[np.int32(r[0])]
             for i, r in enumerate(responses)])

        return labels_predicted


def build_knn_digits(dir_digits, bins_x, bins_y):
    """
    Construct a KNN object from a directory containing digit training images.

    Parameters
    ----------
    dir_digits : string
        Path for the directory containing the digit training images.
    bins_x : int
        Number of x-component bins for `spatial_occupancy()`.
    bins_y : int
        Number of y-component bins for `spatial_occupancy()`.

    Returns
    -------
    KNN
        k-Nearest Neighbour algorithm trained on the digit images provided.

    """
    digits = {
        0: "Zero",
        1: "One",
        2: "Two",
        3: "Three",
        4: "Four",
        5: "Five",
        6: "Six",
        7: "Seven",
        8: "Eight",
        9: "Nine"}

    imgs = dict()
    for d in iter(digits):
        for k in range(5):
            digit_file = os.path.join(dir_digits, f"{digits[d]}{k+1}.jpg")
            imgs[(d, k)] = cv2.imread(digit_file, cv2.IMREAD_COLOR)

    samples = dict()
    for d in iter(digits):
        samples[d] = np.zeros((5, bins_x * bins_y), dtype=np.float32)
        for k in range(5):
            img_gray = cv2.cvtColor(imgs[(d, k)], cv2.COLOR_BGR2GRAY)
            _, img_bin = cv2.threshold(img_gray, 128, 255, cv2.THRESH_OTSU)

            regions = cc_regions(img_bin)
            h, w = img_bin.shape[:2]
            region = min(
                regions, key=lambda r: r.distance((int(w/2), int(h/2))))

            samples[d][k] = np.ravel(region.spatial_occupancy(bins_x, bins_y))

    return KNN(samples)


def build_knn_arrows(dir_arrows, bins_x, bins_y):
    """
    Construct a KNN object from a directory containing arrow training images.

    Parameters
    ----------
    dir_arrows : string
        Path for the directory containing the arrow training images.
    bins_x : int
        Number of x-component bins for `spatial_occupancy()`.
    bins_y : int
        Number of y-component bins for `spatial_occupancy()`.

    Returns
    -------
    KNN
        k-Nearest Neighbour algorithm trained on the arrow images provided.

    """
    arrows = {
        "L": "LeftArrow",
        "R": "RightArrow"}

    imgs = dict()
    for a in iter(arrows):
        for k in range(5):
            arrow_file = os.path.join(dir_arrows, f"{arrows[a]}{k+1}.jpg")
            imgs[(a, k)] = cv2.imread(arrow_file, cv2.IMREAD_COLOR)

    samples = dict()
    for a in iter(arrows):
        samples[a] = np.zeros((5, bins_x * bins_y), dtype=np.float32)
        for k in range(5):
            img_gray = cv2.cvtColor(imgs[(a, k)], cv2.COLOR_BGR2GRAY)
            _, img_bin = cv2.threshold(img_gray, 128, 255, cv2.THRESH_OTSU)

            regions = cc_regions(img_bin)
            h, w = img_bin.shape[:2]
            region = min(
                regions, key=lambda r: r.distance((int(w/2), int(h/2))))

            samples[a][k] = np.ravel(region.spatial_occupancy(bins_x, bins_y))

    return KNN(samples)
