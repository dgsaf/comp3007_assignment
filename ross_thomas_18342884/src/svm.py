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

        for li in self.labels:
            si = samples_labelled[li].astype(np.float32)
            ni = si.shape[0]
            ri = np.array([li for i in range(ni)], dtype=np.short32)

            for lj in self.labels:
                if lj <= li:
                    continue

                sj = samples_labelled[lj].astype(np.float32)
                nj = sj.shape[0]
                rj = np.array([lj for j in range(nj)], dtype=np.short32)

                lij = (li, lj)
                nij = (ni + nj)

                wi = nij / (2 * ni)
                wj = nij / (2 * nj)
                wij = np.array([wi, wj])

                self._svms[lij] = cv2.ml.SVM_create()
                self._svms[lij].setType(cv2.ml.SVM_C_SVC)
                self._svms[lij].setKernel(cv2.ml.SVM_RBF)
                self._svms[lij].setTermCriteria(
                    (cv2.TERM_CRITERIA_MAX_ITER , 100000, 1.0e-6))
                self._svms[lij].setClassWeights(wij)

                sij = np.append(si, sj, axis=0)
                rij = np.append(ri, rj, axis=0)

                self._svms[lij].trainAuto(
                    sij, cv2.ml.ROW_SAMPLE, rij, kFold=min([nij - 1, 10]))
        return

    def predict(self, samples):
        votes = dict()
        for i, _ in enumerate(samples):
            votes[i] = {k : 0 for k in self.labels}

        for lij in iter(self.svms):
            li, lj = lij

            _, responses = self.svms[lij].predict(samples.astype(np.float32))

            for i, response in enumerate(responses):
                votes[i][response] += 1

        labels_predicted = np.array(
            [max(votes[i], key=votes[i].get) for i in iter(votes)])
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

            samples[d][k] = region.hu_moments_regular

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

            samples[d][k] = region.hu_moments_regular

    return SVM_OVO(samples)