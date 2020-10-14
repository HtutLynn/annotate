import numpy as np
import cv2
import pycuda.driver as cuda
import time

from annotate_cascade import CascadeModel

cuda.init()

model = CascadeModel( "yolov4","trt/yolov4-416.trt", "trt/extractor.trt", "trt/digits.trt",0.5, 0.5)

image = cv2.imread("sample/digits.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

_ = model.detect(image, cls=0)

t1 = time.time()
_ = model.detect(image, cls=0)
t2 = time.time()
print("YOLO inference time : {}s".format(t2 - t1))

a = time.time()
boxes, classes, scores = model.detect(image, cls=0)
features = model.extract(image, boxes)
f_date, f_time = model.extract_timestamp(img=image)
b = time.time()
print("Seperate inference time : {}s".format(b - a))

a = time.time()
preds = model.infer(img=image, cls=0)
b = time.time()
print("Infer inference time : {}s".format(b - a))
print(preds['features'].shape)

print("features : {}".format(features.shape))

print("date : {}".format(preds['date']))
print("time : {}".format(preds['time']))
print(preds)

for box in boxes:
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)

cv2.imwrite("unittest_pipeline.jpg", image)
