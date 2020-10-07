import numpy as np
import cv2

# import tracking modules
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.detection import Detection
from .sort.tracker import Tracker

#import counting modules
from .count.counteralgo import Counter
from .count.stats import Stats


__all__ = ['DeepSortCount']


class DeepSortCount(object):
    def __init__(self, doors, max_dist=0.2, min_confidence=0.3, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=50):
        self.min_confidence = min_confidence

        max_cosine_distance = max_dist
        nn_budget = 50
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, Counter(doors), Stats(len(doors)), 
                               max_iou_distance=max_iou_distance, 
                               max_age=max_age, n_init=n_init)

    def update(self, output, img_shape):
        self.height, self.width = img_shape
        
        # Seperate the outputs
        bbox_xyxy   = output['rois']
        confidences = output['scores']
        features    = output['features']
            
        bbox_tlwh = self._xyxy_to_tlwh_batch(bbox_xyxy)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i,conf in enumerate(confidences) if conf>self.min_confidence]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # return tracks and results
        return self.tracker.tracks, self.tracker.stats

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        else:
            bbox_xywh = np.array(bbox_xywh)
            bbox_tlwh = bbox_xywh.copy()
        bbox_tlwh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
        bbox_tlwh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
        return bbox_tlwh
    
    @staticmethod
    def _xyxy_to_tlwh_batch(bbox_xyxy):
        if isinstance(bbox_xyxy, np.ndarray):
            bbox_tlwh = bbox_xyxy.copy()
        bbox_tlwh[:, 2] = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]
        bbox_tlwh[:, 3] = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]

        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),self.width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),self.height-1)
        return x1,y1,x2,y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1,y1,x2,y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2-x1)
        h = int(y2-y1)
        return t,l,w,h