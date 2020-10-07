# convert efficientdet
import sys
import os

import subprocess
import argparse
import time

import numpy as np
import cv2
import torch
import onnx
import onnxruntime
from torchvision.ops.boxes import batched_nms

from efficient.backbone import EfficientDetBackbone
from efficient.efficientdet.utils import BBoxTransform, ClipBoxes
from efficient.utils.utils import preprocess, invert_affine, postprocess, \
                                  filterout, plot_one_box,  STANDARD_COLORS, \
                                  standard_to_bgr, get_index_label

def transform_to_onnx(coef, weight_file, batch_size, n_classes, IN_IMAGE_H, IN_IMAGE_W):
    """
    Transform the pytorch weights file into onnx runtime model
    
    Parameters
    ----------
    coef        : int
                  Compound coefficient for scaling the EfficientDet architecture
    weight_file : str
                  Pretrained file for EfficientDet{} model
    batch_size  : int
                  Batch size must be explicitly mentioned in conversion process and
                  only that batch size will be used as default in inferencing processes 
    n_classes   : int
                  Number of classes
    IN_IMAGE_H  : int
                  Height of the input image, static during inferencing
    IN_IMAGE_W  : int
                  Width of the input image, static during inferencing
    """
    model = EfficientDetBackbone(compound_coef=coef, num_classes=n_classes,  onnx_export=True)
    model.load_state_dict(torch.load(weight_file))
    model.requires_grad_(False)
    model.cuda()
    model.eval()

    # construct dummy data with static batch_size
    x = torch.randn((batch_size, 3, IN_IMAGE_H, IN_IMAGE_W), requires_grad=True).cuda()

    # file name
    onnx_file_name = "EfficientDet{}_{}.onnx".format(coef, batch_size)

    # Export the model
    torch.onnx.export(model,
                      x,
                      onnx_file_name,
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes=None)

    print("Onnx model exporting done!")
    return onnx_file_name


def main(coef, image_path, batch_size, n_classes):
    """
    Main function for creating ONNX model file and test inferencing.

    Parameters
    ----------
    coef        : int
                  coefficient for scaling the efficientdet architecture
    image_path  : str
                  path to the sample image file
    batch_size  : int
                  Static Batch size to be used in inferencing
    n_classes   : int
                  Number of outputs
    """
    weight_path = f'checkpoints/efficientdet-d{coef}.pth'
    input_sizes = [416, 640, 768, 896, 1024, 1280, 1280, 1536]

    # postprocess functions
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    threshold    = 0.4
    iou_threshold= 0.6
    color_list = standard_to_bgr(STANDARD_COLORS)

    obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']

    # Transform the onnx as specified batch size
    # transform_to_onnx(coef=coef, weight_file=weight_path, batch_size=batch_size, 
    #                   n_classes=n_classes, IN_IMAGE_H=input_sizes[coef], IN_IMAGE_W=input_sizes[coef])

    # check if weight file exists or not
    if not os.path.isfile("checkpoints/efficientdet-d{}.pth".format(coef)):
        process = subprocess.Popen(['wget', '-O', 'checkpoints/efficientdet-d{}.pth'.format(coef), 'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d{}.pth'.format(coef)])
        process.communicate()

    # Transform the onnx for demo
    onnx_demo_name = transform_to_onnx(coef=coef, weight_file=weight_path, batch_size=1, 
                                       n_classes=n_classes, IN_IMAGE_H=input_sizes[coef], IN_IMAGE_W=input_sizes[coef])


    ori_img, framed_img, framed_meta = preprocess(image_path, max_size=input_sizes[coef])
    x = torch.stack([torch.from_numpy(fi) for fi in framed_img], 0)
    x = x.to(torch.float32).permute(0, 3, 1, 2)
    x = x.numpy()

    session = onnxruntime.InferenceSession(onnx_demo_name)
    print(session)
    print("The model expects input shape : ", session.get_inputs()[0].shape)
    # Compute
    input_name = session.get_inputs()[0].name
    results = session.run(None, {input_name: x})
    print(results)
    print(len(results))

    print(results[0].shape)
    print(results[1].shape)

    bboxes, classification = results
    # regression     = torch.tensor(regression)
    transformed_anchors = torch.tensor(bboxes)
    transformed_anchors = clipBoxes(transformed_anchors, x)
    classification = torch.tensor(classification)
    # anchors        = torch.tensor(anchors) 

    # out = postprocess(x,
    #                   anchors, regression, classification,
    #                   regressBoxes, clipBoxes, 
    #                   threshold, iou_threshold)

    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]
    out = []
    for i in range(x.shape[0]):
        if scores_over_thresh[i].sum() == 0:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })
            continue

        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        scores_, classes_ = classification_per.max(dim=0)
        anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)

        if anchors_nms_idx.shape[0] != 0:
            classes_ = classes_[anchors_nms_idx]
            scores_ = scores_[anchors_nms_idx]
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]

            out.append({
                'rois': boxes_.cpu().numpy(),
                'class_ids': classes_.cpu().numpy(),
                'scores': scores_.cpu().numpy(),
            })
        else:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })


    out = invert_affine(framed_meta, out)
    out = out[0]
    for j in range(len(out['rois'])):
        x1, y1, x2, y2 = out['rois'][j].astype(np.int)
        obj = obj_list[out['class_ids'][j]]
        score = float(out['scores'][j])
        plot_one_box(ori_img[0], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])

    cv2.imwrite("efficientdet{}_onnx_inferenced.jpg".format(coef), ori_img[0])

if __name__ == '__main__':
    print("Converting to onnx and running demo ...")
    if len(sys.argv) == 5:

        coef        = int(sys.argv[1])
        image_path  = sys.argv[2]
        batch_size  = int(sys.argv[3])
        n_classes   = int(sys.argv[4])

        main(coef, image_path, batch_size, n_classes) 

    else:
        print("Please run this way:\n")
        print(" print('  python demo_onnx.py <compound_coefficient> <image_path> <batch_size> <n_classes>')")

# python convert_onnx_efficientdet.py 1 221.jpg 1 90