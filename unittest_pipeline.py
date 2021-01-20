import cv2
import pycuda.driver as cuda
import time

# import model
from cascade import CascadeModel

cuda.init()

model = CascadeModel("trt/yolov4-416.trt", "trt/extractor.trt", 0.5, 0.5, 80)

image = cv2.imread("sample/221.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

_ = model.detect(image)

t1 = time.time()
_ = model.detect(image)
t2 = time.time()
print("YOLO inference time : {}s".format(t2 - t1))

a = time.time()
boxes, classes, scores = model.detect(image)
features = model.extract(image, boxes, classes, scores)
b = time.time()
print("Seperate inference time : {}s".format(b - a))

a = time.time()
preds = model.infer(img=image, class_id=0)
b = time.time()
print("Infer inference time : {}s".format(b - a))
print(preds['features'].shape)

print("features : {}".format(features.shape))

for box in boxes:
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)

cv2.imwrite("unittest_pipeline.jpg", image)
