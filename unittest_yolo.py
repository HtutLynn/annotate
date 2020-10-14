import numpy as np
import cv2
import time

import pycuda.autoinit
from yolo import TrtYOLOv4

model = TrtYOLOv4(engine_path="trt/yolov4-416.trt", input_shape=(416, 416), nms_thres=0.5, conf_thres=0.5)

image = cv2.imread("sample/221.jpg")

# test run
_ = model.detect(img=image)

t0 = time.time()
boxes, _, _ = model.detect(image)
t1 = time.time()

for box in boxes:
    x1, y1, x2, y2 = box
    image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)

print("Inferred time : {}".format(t1 - t0))

cv2.imwrite("unittest_yolo.jpg", image)