# import tensorrt related libraries
import pycuda.autoinit

# import system libraries
import argparse
import os
import cv2
import numpy as np
import time

# import custom libraries
from utils.yamlparser import YamlParser
from utils.blur import blur_bboxes

# import core components
from deepsortcount.deep_sort_count import DeepSortCount
from deepsortcount.count.door import DoorInfo
from cascade import CascadeModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="demo.mp4")
    parser.add_argument("--model", type=str, default="efficientdet")
    parser.add_argument("--detect_engine", type=str, default="trt/efficientdet-d1.trt")
    parser.add_argument("--featext_engine", type=str, default="trt/extractor.trt")
    parser.add_argument("--conf_threshold", type=float, default=0.5)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument("--config_doors", type=str, default="./configs/scene2.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort_count.yaml")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--save_path", type=str, default="./outputs/")
    parser.add_argument("--blur", type=bool, default=False)
    parser.add_argument("--record", type=bool, default=True)
    parser.add_argument("--display", type=bool, default=True)
    return parser.parse_args()

def build_deepsortcount_tracker(cfg, doors):

    return DeepSortCount(doors=doors,
                         max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                         max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE, max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT,
                         nn_budget=cfg.DEEPSORT.NN_BUDGET)

def build_doors(cfg):
    """
    Build DoorInfo instances for recording human activity for each door
    """
    doors = []

    for count in range(len(cfg)):
        door_key = "Door" + str(count)
        cur_door = DoorInfo(first_checkpoint_box=cfg[door_key].firstcheckpoint,
                            second_checkpoint_box=cfg[door_key].secondcheckpoint)
        doors.append(cur_door)

    return doors

def visualize(image, preds, tracks, doors, stats, blur, time):
    """
    Vsiualize the inference results
    """
    # show inference info
    fps_text = "FPS : {:.2f}".format(1/time)
    cv2.putText(image, fps_text, (11, 40), cv2.FONT_HERSHEY_PLAIN, 4.0, (32, 32, 32), 4, cv2.LINE_AA)

    for i, door in enumerate(doors):
        coor1 = np.array(door.first_checkpoint_box, np.int32)
        coor2 = np.array(door.second_checkpoint_box, np.int32)
        door = "Door {} : {}".format(i, stats.Doors[i])
        cv2.rectangle(image, (coor1[0], coor1[1]), (coor1[2], coor1[3]), (0, 255, 0), 2)
        cv2.rectangle(image, (coor2[0], coor2[1]), (coor2[2], coor2[3]), (0, 255, 0), 2)
        cv2.putText(image, door, (coor2[0] - 20, coor2[1] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 2)

    
    if preds:
        if blur:
            image = blur_bboxes(image=image, bboxes=preds['rois'])

        else:
            for box in range(len(preds['rois'])):
                (x1, y1, x2, y2) = preds['rois'][box].astype(np.int)
                score = float(preds['scores'][box])
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(image, '{:.3f}'.format(score),
                            (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 0), 1)
        
        for track in tracks:

            if track.is_tentative():
                obj = track.track_id
                x_mid, y_mid = track.centroid
                cv2.circle(image, (x_mid , y_mid), 4, (255, 255, 154), -1)

                cv2.putText(image, str(obj),
                            (x_mid - 10, y_mid - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 154), 1)

            elif track.is_confirmed() and track.time_since_update == 0:
                obj = track.track_id

                x_mid, y_mid = track.centroid
                cv2.circle(image, (x_mid , y_mid), 4, (255, 255, 0), -1)
                cv2.putText(image, str(obj),
                            (x_mid - 10, y_mid - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 0), 1)
            
            elif track.is_confirmed() and track.time_since_update > 0:
                obj = track.track_id
                x_mid, y_mid = track.centroid
                cv2.circle(image, (x_mid , y_mid), 4, (128, 0, 128), -1)

                cv2.putText(image, str(obj),
                            (x_mid - 10, y_mid - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (128, 0, 128), 1)

    return image


class CascadePipeline(object):
    """
    Computer Vision Pipeline where TensorRT models run in a cascade manner.
    """

    def __init__(self, cfg):
        """Setup parameters required for the entire Cascade CV Pipeline

        Parameters
        ----------
        cfg     : argument instance
                  Dictionary-like argument class, containing infos about pipeline

        Attributes
        ----------
        video           : str
                          Path of the video file, to be inferred
        yolov4_engine   : str
                          Path of the YOLOv4 TensorRT engine file
        featext_engine  : str
                          Path of the Feature Extractor TensorRT engine file
        conf_threshold  : int
                          Threshold value for confidence
        nms_threshold   : int
                          Threshold value for non-maximum suppression
        config_doors    : str
                          Path of the Yaml config file of the doors in a scene
        config_deepsort : str
                          Path of the Yaml config file of DeepSort Tracker
        save_path       : str
                          path where inferred video is to be saved at
        blur            : boolean
                          Whether to blur the detections for privacy or not
        record          : boolean
                          Whether to record the inferred results or not    
        display         : boolean
                          Whether to display the inferred results or not    
        """

        self.model               = cfg.model
        self.detect_engine_path  = cfg.detect_engine
        self.featext_engine_path = cfg.featext_engine

        self.conf_threshold      = cfg.conf_threshold
        self.nms_threshold       = cfg.nms_threshold

        # setup doors
        self.doors_cfg  = YamlParser(cfg.config_doors)
        self.doors      = build_doors(self.doors_cfg)

        # build the cascade model
        self.cascademodel = CascadeModel(model=self.model, detect_engine_path=self.detect_engine_path, featext_engine_path=self.featext_engine_path,
                                         nms_thres=self.nms_threshold, conf_thres=self.conf_threshold)

        # setup parameters for deepsort tracking algorithm
        self.track_cfg = YamlParser(cfg.config_deepsort)
        self.deepsortcount = build_deepsortcount_tracker(cfg=self.track_cfg, doors=self.doors)

        # setup io files parameters
        self.video_path = cfg.video
        self.blur = cfg.blur
        self.record = cfg.record
        self.display = cfg.display

        self.WINDOW_NAME = "Cascade-TensorRT-Pipeline"
        title = "Detect TensorRT + FeatureExtractor TensorRT + DeepSort + Count"
        self._set_window(self.WINDOW_NAME, title)

        if self.record:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            path = path = cfg.save_path + "/videos/" + "cascade_TensorRT_pipeline.mp4"
            self.writer = cv2.VideoWriter(path, fourcc, 20, (self.im_width, self.im_height), True)

    def _set_window(self, window_name, title):
        """Set the width and height of the video if self.record is True
        """

        assert os.path.isfile(self.video_path), "Path error"
        vc = cv2.VideoCapture()
        vc.open(self.video_path)
        self.im_width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.im_height, self.im_height)
        cv2.setWindowTitle(window_name, title)

    def run(self):
        """
        Run the Cascade TensorRT Computer Vision Pipeline
        """
        # first get started with by readind frames from video
        cap = cv2.VideoCapture(self.video_path)
        while True:
            if cv2.getWindowProperty(self.WINDOW_NAME, 0) < 0:
                break

            ret, frame = cap.read()
            if not ret:
                break
            start_time = time.time()
            # do cascade inference
            infer_time0 = time.time()
            preds = self.cascademodel.infer(frame, cls=0)
            infer_time1 = time.time()

            # track with deepsort tracker.
            track_time0 = time.time()
            tracks, stats = self.deepsortcount.update(preds, frame.shape[:2])
            track_time1 = time.time()

            end_time = time.time()

            total_time = end_time - start_time

            if self.display:
                image = visualize(image=frame, preds=preds, tracks=tracks,
                                  doors=self.doors, stats=stats, blur=self.blur, time=total_time)

                cv2.imshow(self.WINDOW_NAME, image)
            
            if self.record:
                self.writer.write(image)

            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_args()
    pipeline = CascadePipeline(args)
    pipeline.run()