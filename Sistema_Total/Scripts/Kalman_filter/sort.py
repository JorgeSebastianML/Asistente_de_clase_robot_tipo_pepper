"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

from numba import jit
import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from sklearn.utils.linear_assignment_ import linear_assignment
import glob
import time
import argparse
from filterpy.kalman import KalmanFilter


@jit
def iou(bb_test, bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return (o)


def colinearity(det, hist):
    '''
    det - current detection
    hist - last 2 mean detections
    '''
    dims = det[2:4] - det[:2]
    diag = np.sqrt(sum(dims ** 2))
    a = det[:2] + dims / 2 - hist[-2]
    b = hist[-1] - hist[-2]
    len1 = np.sqrt(sum(a * a))
    len2 = np.sqrt(sum(b * b))
    ratio = len2 / float(len1)
    maxdist = diag * (min(dims) / max(dims) + 1)
    maxval = b.dot(b)
    a *= ratio
    return a.dot(b) / float(maxval) if maxval and maxdist > len1 else 0


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.cthist = [self.kf.x[:2].ravel()]

    def update(self, bbox, n):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        self.cthist.append(bbox[:2] + (bbox[2:4] - bbox[:2]) / 2)
        self.cthist = self.cthist[-n:]

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
            self.kf.P *= 1.2  # we may be lost, increase uncertainty and responsiveness
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, cost_fn=iou, threshold=0.33):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    lendet = len(detections)
    lentrk = len(trackers)

    if (lentrk == 0):
        return np.empty((0, 2), dtype=int), np.arange(lendet), np.array([], dtype=int)
    cost_matrix = np.zeros((lendet, lentrk), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            cost_matrix[d, t] = cost_fn(det, trk)
    cost_matrix[cost_matrix < threshold] = 0.
    matched_indices = linear_assignment(-cost_matrix)

    costs = cost_matrix[tuple(matched_indices.T)]  # select values from cost matrix by matched indices
    matches = matched_indices[np.where(costs)[0]]  # remove zero values from matches
    unmatched_detections = np.where(np.in1d(range(lendet), matches[:, 0], invert=True))[0]
    unmatched_trackers = np.where(np.in1d(range(lentrk), matches[:, 1], invert=True))[0]

    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)

    return matches, unmatched_detections, unmatched_trackers


class Sort(object):
    def __init__(self, max_age=10, min_hits=0):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0

    def update(self, dets, cnum=3):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
          cnum - number of center positions to average
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        ctmean = []
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if (np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks)

        for t in unmatched_trks:
            cnt = np.array(self.trackers[t].cthist)
            cnt = np.array([np.convolve(cnt[:, i], np.ones((cnum,)) / cnum, mode='valid') for i in (0, 1)]).T
            if cnt.shape[0] == 1:  # fix same len
                cnt = np.concatenate((cnt, cnt), axis=0)
            ctmean.append(cnt)

        rematch, new_dets, lost_trks = associate_detections_to_trackers(dets[unmatched_dets], ctmean, colinearity, 0.6)
        rematch = np.array([unmatched_dets[rematch[:, 0]], unmatched_trks[rematch[:, 1]]]).T
        matched = np.concatenate((matched, rematch.reshape(-1, 2)))
        unmatched_dets = unmatched_dets[new_dets]
        unmatched_trks = unmatched_trks[lost_trks]

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if (t not in unmatched_trks):
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d, :][0], cnum + 1)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if ((trk.time_since_update < self.max_age) and (
                    trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                ret.append(np.concatenate((d, [trk.id + 1], [trk.time_since_update])).reshape(1,
                                                                                              -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))



