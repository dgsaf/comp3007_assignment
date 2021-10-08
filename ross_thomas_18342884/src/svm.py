#!/usr/bin/env python

import os
import numpy as np
import cv2

from region import *

class SVM_OVO:
    def __init__(self, samples_labelled):
        self.train(samples_labelled)
        return

    @property
    def labels(self):
        return self._labels

    @property
    def svms(self):
        return self._svms

    def train(self, samples_labelled):
        # self._labels = set(iter(samples_labelled))
        self._labels = {k : l for k, l in enumerate(samples_labelled.keys())}

        self._svms = dict()

        for ki in self.labels:
            si = samples_labelled[ki].astype(np.float32)
            ni = si.shape[0]
            ri = np.array([ki for i in range(ni)], dtype=np.int32)

            for kj in self.labels:
                if kj <= ki:
                    continue

                sj = samples_labelled[kj].astype(np.float32)
                nj = sj.shape[0]
                rj = np.array([kj for j in range(nj)], dtype=np.int32)

                kij = (ki, kj)
                nij = (ni + nj)

                wi = nij / (2 * ni)
                wj = nij / (2 * nj)
                wij = np.array([wi, wj])

                self._svms[kij] = cv2.ml.SVM_create()
                self._svms[kij].setType(cv2.ml.SVM_C_SVC)
                self._svms[kij].setKernel(cv2.ml.SVM_RBF)
                self._svms[kij].setTermCriteria(
                    (cv2.TERM_CRITERIA_MAX_ITER , 100000, 1.0e-6))
                self._svms[kij].setClassWeights(wij)

                sij = np.append(si, sj, axis=0)
                rij = np.append(ri, rj, axis=0)

                self._svms[kij].trainAuto(
                    sij, cv2.ml.ROW_SAMPLE, rij, kFold=min([nij, 10]))
        return

    def predict(self, samples):
        votes = dict()
        for i, sample in enumerate(samples):
            votes[i] = {k : 0 for k in self.labels}

        for kij in iter(self.svms):
            ki, kj = kij

            _, responses = self.svms[kij].predict(samples.astype(np.float32))

            for i, response in enumerate(responses):
                k = np.int32(response[0])
                votes[i][k] += 1

        labels_predicted = np.array(
            [self.labels[max(votes[i], key=votes[i].get)] for i in iter(votes)])
        return labels_predicted


def build_svm_digits(dir_digits):
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
        samples[d] = np.zeros((5, 7), dtype=np.float32)
        for k in range(5):
            img_gray = cv2.cvtColor(imgs[(d, k)], cv2.COLOR_BGR2GRAY)
            _, img_bin = cv2.threshold(img_gray, 128, 255, cv2.THRESH_OTSU)

            regions = cc_regions(img_bin)
            h, w = img_bin.shape[:2]
            region = min(
                regions, key=lambda r: r.distance((int(w/2), int(h/2))))

            samples[d][k] = region.features

    return SVM_OVO(samples)


def build_svm_arrows(dir_arrows):
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
        samples[a] = np.zeros((5, 7), dtype=np.float32)
        for k in range(5):
            img_gray = cv2.cvtColor(imgs[(a, k)], cv2.COLOR_BGR2GRAY)
            _, img_bin = cv2.threshold(img_gray, 128, 255, cv2.THRESH_OTSU)

            regions = cc_regions(img_bin)
            h, w = img_bin.shape[:2]
            region = min(
                regions, key=lambda r: r.distance((int(w/2), int(h/2))))

            samples[d][k] = region.features

    return SVM_OVO(samples)
