"""annotate_async.py

This is the 'async' version of annotate.py. It creates
dedicated child thread for fetching camera input and do inferencing
with the TensorRT optimized Cascade model/engine, and another child thread
for performing tracking with DeepSORT tracker, while using the main
thread for drawing detection results and displaying video. Ideally,
the 3 threads work in a pipeline fashion so overall throughput (FPS)
would be improved comparing to the non-async version. Another TensorRT
model, digits-recognizers is used to extract timestamp information
from the parsed cctv video data.
"""

# import tensorrt related libraries
import pycuda.driver as cuda
import tensorrt as trt

# import system libraries
import os
import cv2
import csv
import time
import argparse
import threading
import numpy as np

# import custom libraries
from utils.blur import blur_bboxes
from utils.yamlparser import YamlParser

# import core components
from annotate_cascade import CascadeModel
from deepsortcount.count.door import DoorInfo
from deepsortcount.deep_sort_count import DeepSortCount

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="demo.mp4")
    parser.add_argument("--model", type=str, default="efficientdet")
    parser.add_argument("--detect_engine", type=str, default="trt/efficientdet-d1.trt")
    parser.add_argument("--featext_engine", type=str, default="trt/extractor.trt")
    parser.add_argument("--digits_engine", type=str, default="trt/digits.trt")
    parser.add_argument("--conf_threshold", type=float, default=0.5)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument("--config_doors", type=str, default="./configs/scene2.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort_count.yaml")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--save_path", type=str, default="./outputs/")
    parser.add_argument("--blur", type=bool, default=False)
    parser.add_argument("--record", type=bool, default=True)
    parser.add_argument("--annotate", type=bool, default=True)
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

def visualize(image, preds, tracks, doors, stats, blur, fps):
    """
    Vsiualize the inference results
    """
    # show inference info
    fps_text = "FPS : {:.2f}".format(fps)
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

class CascadeTrtThread(threading.Thread):
    """Cascade model TensorRT child thread

    This implements the child thread which continues to read images
    from cam(input) and to do Cascade TensorRT engine inferencing.
    The child thread stores and the input image and inferred results
    into global variables, and uses a condition varaiable to inform
    main thread. In other words, this thread acts as the producer 
    while thread is the consumer.
    """

    def __init__(self, condition, cfg, cam):
        """
        Setup parameters required Cascade model TensorRT engine

        Parameters
        ----------
        condition : Threading condition
                    The condition variable, used to notify main
                    thread about new frame and detection result,
        cam       : The camera object for reading input image frames

        cfg       : argument parser
                    Dictionary like object, containing all the required
                    informations

        Attributes
        ----------
        model           : str
                          Type of the Detection model architecture, YOLOv4 or EfficientDet
        detect_engine   : str
                          Path of the Human Detection TensorRT engine file
        featext_engine  : str
                          Path of the Feature Extractor TensorRT engine file
        conf_threshold  : int
                          Threshold value for confidence
        nms_threshold   : int
                          Threshold value for non-maximum suppression
        obj_list        : list
                          List, containing label names of model
        """

        threading.Thread.__init__(self)
        self.condition           = condition
        self.model               = cfg.model
        self.detect_engine_path  = cfg.detect_engine
        self.featext_engine_path = cfg.featext_engine
        self.digits_engine_path  = cfg.digits_engine

        self.conf_threshold      = cfg.conf_threshold
        self.nms_threshold       = cfg.nms_threshold

        # Threading attributes
        # NOTE: cuda_ctx code has been moved into Cascade model TensorRT class
        #self.cuda_ctx = None  # to be created when run

        self.cam = cam
        self.running = False

    def run(self):
        """Run until 'running' flag is set to False by main thread

        NOTE: CUDA context is created here, i.e. inside the thread
        which calls CUDA kernels.  In other words, creating CUDA
        context in __init__() doesn't work.
        """

        global image, preds, write_flag

        print("CascadeTrtThread: Loading the Engine...")

        # build the cascade TensorRT model engine
        self.cascademodel = CascadeModel(model=self.model, detect_engine_path=self.detect_engine_path, featext_engine_path=self.featext_engine_path,
                                         digits_engine_path=self.digits_engine_path,
                                         nms_thres=self.nms_threshold, conf_thres=self.conf_threshold)

        
        print("CascadeTrtThread: start running...")
        self.running = True
        while self.running:
            ret, frame = self.cam.read()

            if ret:
                results = self.cascademodel.infer(frame, cls=0)
                with self.condition:
                    preds      = results
                    image      = frame
                    write_flag = True
                    self.condition.notify()
            else:
                with self.condition:
                    write_flag = False
                    self.condition.notify()

        # delete the model after inference process
        del self.cascademodel

        print("CascadeTrtThread: stopped...")

    def stop(self):
        self.running = False
        self.join()

def _set_window(video_path,  window_name, title):
    """Set the width and height of the video if self.record is True
    """

    assert os.path.isfile(video_path), "Path error"
    vc = cv2.VideoCapture()
    vc.open(video_path)
    im_width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    im_height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return (im_width, im_height)

def track_annotate_and_display(condition, cfg, input_size, name):
    """
    1. Track and count humans with DeepSort Tracker.
    2. Display the result

    Parameters
    ----------
    condition : Threading condition
                Used to parse value from TensorRT thread
    cfg       : argument parser
                Dictionary like object, containing all the required
                informations
    """

    global image, preds, write_flag

    # parse data from arguments
    input_size = input_size
    blur       = cfg.blur
    record     = cfg.record
    annotate_flag   = cfg.annotate

    # build deepsortcount tracker
    # fristly build doors
    doors_cfg = YamlParser(cfg.config_doors)
    doors     = build_doors(cfg=doors_cfg)

    # finally, build doors
    # setup parameters for deepsort tracking algorithm
    track_cfg = YamlParser(cfg.config_deepsort)
    deepsortcount = build_deepsortcount_tracker(cfg=track_cfg, doors=doors)

    annotate_data = []

    _file_name = os.path.abspath(cfg.video).split("/")[-1].split('.')[0]
    video_name = _file_name + "_annotated.mp4"
    annotation_name = _file_name + "_annotation.csv"


    # writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    path = cfg.save_path + video_name
    writer = cv2.VideoWriter(path, fourcc, 20, (input_size[0], input_size[1]), True)

    fps = 0.0
    tic = time.time()
    while True:
        with condition:
            # Wait for the next frame and detection result.  When
            # getting the signal from the child thread, save the
            # references to the frame and detection result for
            # deepsortcount tracker. And then display.
            condition.wait()
            frame, results, flag = image, preds, write_flag

        if flag:
            tracks, stats = deepsortcount.update(output=results, img_shape=input_size)
            frame_date          = results['date']
            frame_time          = results['time']

            if annotate_flag:
                annotation = [frame_date, frame_time]
                for door_no in range(len(doors)):
                    annotation.append(stats.Doors[door_no])

                # print(annotate_data)
                annotate_data.append(annotation)

            if record:
                drawn_image = visualize(image=frame, preds=results, tracks=tracks,
                                    doors=doors, stats=stats, blur=cfg.blur, fps=fps)
                writer.write(drawn_image)

            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
            tic = toc
        else:
            break
        
    if annotate_flag:
        with open("annotation/" + annotation_name, "w", encoding="utf=8") as WR:
            writer = csv.writer(WR)
            for row in annotate_data:
                writer.writerow(row) 

def main():
    """Main Thread

    Attributes
    ----------
    video           : str
                      Path of the video file, to be inferred
                      
    """
    # initiate cuda
    cuda.init()

    # parse arguments
    args = parse_args()
    print(args)
    video = args.video

    # set cv2 window
    WINDOW_NAME = "Cascade-TensorRT_async-Pipeline"
    title = "Detector TensorRT + FeatureExtractor TensorRT + DeepSort + Count"
    input_size = _set_window(video, WINDOW_NAME, title)

    # open a videofile, using cv2

    cam = cv2.VideoCapture(video)

    # threading condition
    condition  = threading.Condition()
    trt_thread = CascadeTrtThread(condition, args, cam)
    trt_thread.start()
    track_annotate_and_display(condition, args, input_size, WINDOW_NAME)
    trt_thread.stop()

    cam.release()

if __name__ == "__main__":
    main()