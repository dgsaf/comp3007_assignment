#!/usr/bin/env python

import numpy as np
import cv2


class SvmOVO:
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
        self._labels = set(iter(samples_labelled))
        self._svms = dict()

        for li in self.labels:
            si = samples_labelled[li].astype(np.float32)
            ni = si.shape[0]
            ri = li * np.ones((ni))

            for lj in self.labels:
                if lj < li:
                    continue

                sj = samples_labelled[lj].astype(np.float32)
                nj = sj.shape[0]
                rj = lj * np.ones((nj))

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
            votes[i] = {l : 0 for l in self.labels}

        for lij in iter(self.svms):
            li, lj = lij

            _, responses = self.svms[lij].predict(samples)

            for i, response in enumerate(responses):
                votes[i][response] += 1

        labels_predicted = np.array(
            [max(votes[i], key=votes[i].get) for i in iter(votes)])
        return labels_predicted
