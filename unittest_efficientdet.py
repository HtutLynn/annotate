import numpy as np
import cv2
import time
import pycuda.autoinit

from efficientdet import EfficientDetModel

model = EfficientDetModel(engine_path='checkpoints/efficientdet-b1.trt', nms_thres=0.5,
                          conf_thres=0.3)

image = cv2.imread("221.jpg")

# test run
_ = model.detect(image, cls=0)

t0 = time.time()
boxes, _, _ = model.detect(image, cls=0)
t1 = time.time()

for box in boxes:
    print(box)
    x1, y1, x2, y2 = box
    image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)

print("Inferred time : {}".format(t1 - t0))

cv2.imwrite("unittest_efficientdet.jpg", image)