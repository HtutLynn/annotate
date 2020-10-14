from __future__ import print_function
from os import stat
from typing import List, Tuple, Union

import cv2
import time
import numpy as np
import os.path as ops
import tensorrt as trt
import pycuda.driver as cuda

def _preprocess_fe(img, input_shape, mean, std):
    """Preprocess an image before feature extractor TensorRT model inferencing.

    Parameters
    ----------

    img         : numpy array
                  In RGB format
                  int8 numpy array of shape (img_h, img_w, 3)
    input_shape : tuple
                  a tuple of (H, W)
    
    Returns
    -------
    img         : numpy array
                  preprocessed image
                  float32 numpy array of shape (3, H, W)
    """

    
    # reshape the image
    img  = cv2.resize(img, (input_shape[1], input_shape[0]))

    # transpose the image to pytorch format since the model is converted from pytorch model
    img =  np.transpose(img, (2, 0, 1)).astype(np.float32)

    # unsqueeze extra dimension
    img = np.expand_dims(img, axis=0)

    # squeeze the valueof each pixel to 0 - 1 range
    img /= 255.0

    # perform normalization per channel
    img[:, 0, :, :] = img[:, 0, :, :] - mean[0] / std[0]
    img[:, 1, :, :] = img[:, 1, :, :] - mean[1] / std[1]
    img[:, 2, :, :] = img[:, 2, :, :] - mean[2] / std[2]

    return img

class DigitsPreprocess(object):
    """
    Wrapper class for pre-processing functions for Digits TensorRT runtime model
    """
    def __init__(self, width, height, interpolation=None):
        """
        Initialize parameters required for  pre-processing
        """
        self.width = int(width)
        self.height = int(height)
        self.interpolation = interpolation

    def _preprocess_digit_crop(self, digit_crop):
        """
        Do preprocess on one digit crop
        
        Parameters
        ----------
        digit_crop : numpy array
                     Image numpy array, cropped from the frame
        """
        resized_image = cv2.resize(digit_crop, (self.height, self.width))
        preprocessed_image = (resized_image/255.0 - 0.1307) / 0.3081
        # preprocessed_image = preprocessed_image.transpose((2, 0, 1)).astype(np.float32)
        
        return preprocessed_image

    def __call__(self, image):
        """
        Perform pre-processing on input frame and create batch image 
        for extracting timestamp.

        Parameters
        ----------
        image : numpy array
                Image numpy array, parsed from cctv video data
        """

        # convert BGR image to grayscale image
        grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        batch_img = np.zeros((14, 1, self.height, self.width), dtype=np.float32)

        batch_img[0] = self._preprocess_digit_crop(grayscale_img[56:97, 43:72].copy())
        batch_img[1] = self._preprocess_digit_crop(grayscale_img[56:97, 71:95].copy())
        batch_img[2] = self._preprocess_digit_crop(grayscale_img[56:97, 120:143].copy())
        batch_img[3] = self._preprocess_digit_crop(grayscale_img[56:97, 141:167].copy())
        batch_img[4] = self._preprocess_digit_crop(grayscale_img[56:97, 191:215].copy())
        batch_img[5] = self._preprocess_digit_crop(grayscale_img[56:97, 214:240].copy())
        batch_img[6] = self._preprocess_digit_crop(grayscale_img[56:97, 238:264].copy())
        batch_img[7] = self._preprocess_digit_crop(grayscale_img[56:97, 262:289].copy())
        batch_img[8] = self._preprocess_digit_crop(grayscale_img[56:97, 405:434].copy())
        batch_img[9] = self._preprocess_digit_crop(grayscale_img[56:97, 430:458].copy())
        batch_img[10] = self._preprocess_digit_crop(grayscale_img[56:97, 476:504].copy())
        batch_img[11] = self._preprocess_digit_crop(grayscale_img[56:97, 502:529].copy())
        batch_img[12] = self._preprocess_digit_crop(grayscale_img[56:97, 548:576].copy())
        batch_img[13] = self._preprocess_digit_crop(grayscale_img[56:97, 573:603].copy())

        return batch_img

class YOLOPreprocess(object):
    """
    Wrapper class for pre-processing functions for YOLOv4 TensorRT runtime model
    """
    def __call__(self, img, input_shape):
        """Preprocess an image before YOLOv4 TensorRT model inferencing.

        Parameters
        ----------
        img         : numpy array
                    In RGB format
                    int8 numpy array of shape (img_h, img_w, 3)
        input_shape : tuple
                    a tuple of (H, W)

        Returns
        -------
        img         : numpy array
                    preprocessed image
                    float32 numpy array of shape (3, H, W)

        """

        img = cv2.resize(img, (input_shape[1], input_shape[0]))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)).astype(np.float32)
        img /= 255.0
        return img

class EfficientDetPreprocess(object):

    def __init__(self, mean, std, interpolation=None):
        """
        Initialize parameters for preprocessing

        Parameters
        ----------
        mean : tuple
               tuple, containing values for each channel for normalization
        std  : tuple
               tuple, containing values for each channel for normalization
        """
        self.mean = mean
        self.std  = std
        self.interpolation = interpolation
    
    @staticmethod
    def _aspectaware_resize_padding(image, width, height, interpolation=None, means=None):
        """
        Pads input image without losing the aspect ratio of the original image

        Parameters
        ----------
        image         : numpy array
                        In BGR format
                        uint8 numpy array of shape (img_h, img_w, 3)
        width         : int
                        width of newly padded image
        height        : int
                        height of newly padded image
        interpolation : str
                        method, to be applied on the image for resizing
        
        Returns
        -------       
        canvas        : numpy array
                        float 32 numpy array of shape (height, width, 3)
        new_w         : int
                        width, of the image after resizing without losing aspect ratio
        new_h         : int
                        height, of the image after resizing without losing aspect ratio
        old_w         : int
                        width, of the image before padding
        old_h         : int
                        height, of the image before padding
        padding_w     : int
                        width, of the image after padding
        padding_h     : int
                        height, of the image after padding
        """
        old_h, old_w, c = image.shape
        if old_w > old_h:
            new_w = width
            new_h = int(width / old_w * old_h)
        else:
            new_w = int(height / old_h * old_w)
            new_h = height

        canvas = np.zeros((height, height, c), np.float32)
        if means is not None:
            canvas[...] = means

        if new_w != old_w or new_h != old_h:
            if interpolation is None:
                image = cv2.resize(image, (new_w, new_h))
            else:
                image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

        padding_h = height - new_h
        padding_w = width - new_w

        if c > 1:
            canvas[:new_h, :new_w] = image
        else:
            if len(image.shape) == 2:
                canvas[:new_h, :new_w, 0] = image
            else:
                canvas[:new_h, :new_w] = image

        return canvas, new_w, new_h, old_w, old_h, padding_w, padding_h,

    def __call__(self, img, input_shape):
        """
        Preprocess an image for EfficientDet TensorRT model inferencing.

        Parameters
        ----------
        img         : numpy array
                    In BGR format
                    uint8 numpy array of shape (img_h, img_w, 3)
        input_shape : tuple
                    a tuple of (H, W)
        mean        : tuple
                    tuple of mean value for each dimension
        std         : tuple
                    tuple of std value for each dimension

        Returns
        -------
        img         : numpy array
                    preprocessed image
                    float32 numpy array of shape (3, H, W)
        """ 

        normalized_img = (img / 255 - self.mean) / self.std

        if isinstance(input_shape, List) or isinstance(input_shape, Tuple):
            assert input_shape[0] == input_shape[1] , "Input weight and width are not the same."
            input_shape = int(input_shape[0])

        # This is where the freaking errors lie
        # I kinda normalized the input images but didn't pass the normalized_img to padding function
        # therefore, pixels distribution is a mess so that's why model was generating garbage values
        img_meta = self._aspectaware_resize_padding(image=normalized_img, width=input_shape, height=input_shape,
                                                interpolation=self.interpolation, means=None)
        img = img_meta[0].transpose((2, 0, 1)).astype(np.float32)
        # img = img[np.newaxis, ...].astype(np.float32)

        return img, img_meta[1:]

class YOLOv4Postprocess(object):
    """Class for post-processing the three output tensors from YOLOv4."""

    def __init__(self, yolo_masks, yolo_anchors,conf_threshold, 
                 nms_threshold, yolo_input_resolution, num_classes):
        """Initialize with all values that will be kept when processing
        several frames.  Assuming 3 outputs of the network in the case
        of (large) YOLOv, or 2 for the Tiny YOLOv4.

        Parameters
        ----------
        yolo_masks          : list
                              a list of 3 (or 2) three-dimensional tuples for the YOLOv4 masks
        yolo_anchors        : list 
                              a list of 9 (or 6) two-dimensional tuples for the YOLOv4 anchors
        conf_threshold      : float 
                              threshold for object coverage, float value between 0 and 1
        nms_threshold       : float
                              threshold for non-max suppression algorithm,
                              float value between 0 and 1
        input_wh            : tuple
                              tuple (W, H) for the target network
        num_classes         : int
                              number of output categories/classes
        """
        self.masks          = yolo_masks
        self.anchors        = yolo_anchors
        self.conf_threshold = conf_threshold
        self.nms_threshold  = nms_threshold
        self.input_wh       = (yolo_input_resolution[1], yolo_input_resolution[0])
        self.num_classes    = num_classes

    def process(self, outputs, resolution_raw, conf_thres):
        """Take the YOLOv4 outputs generated from a TensorRT forward pass, post-process them
        and return a list of bounding boxes for detected object together with their category
        and their confidences in separate lists.
        
        Parameters
        ----------
        outputs         : list
                          outputs from a TensorRT engine in NCHW format
        resoluition_raw : tuple
                          the original spatial resolution from the input PIL image in WH order
        conf_thres      : float
                          confidence threshold, e.g. 0.3
        """
        outputs_reshaped = []
        for output in outputs:
            outputs_reshaped.append(self._reshape_output(output))

        boxes_xywh, categories, confidences = self._process_yolo_output(outputs_reshaped, resolution_raw, conf_thres)
        
        if len(boxes_xywh) > 0:
            # convert (x, y, width, height) to (x1, y1, x2, y2)
            img_w, img_h = resolution_raw
            xx = boxes_xywh[:, 0].reshape(-1, 1)
            yy = boxes_xywh[:, 1].reshape(-1, 1)
            ww = boxes_xywh[:, 2].reshape(-1, 1)
            hh = boxes_xywh[:, 3].reshape(-1, 1)
            boxes = np.concatenate([xx, yy, xx+ww, yy+hh], axis=1) + 0.5
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0., float(img_w-1))
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0., float(img_h-1))
            boxes = boxes.astype(np.int)
        else:
            boxes = np.zeros((0, 4), dtype=np.int)  # empty

        return boxes, categories, confidences

    def _reshape_output(self, output):
        """Reshape a TensorRT output from NCHW to NHWC format (with expected C=255),
        and then return it in (height,width,3,85) dimensionality after further reshaping.
        
        Parameters
        ----------
        output      : list
                      an output from a TensorRT enginer after inference
        """
        output = np.transpose(output, [0, 2, 3, 1])
        _, height, width, _ = output.shape
        dim1, dim2 = height, width
        dim3 = 3
        # There are num_classes=80 object categories:
        dim4 = (4 + 1 + self.num_classes)
        return np.reshape(output, (dim1, dim2, dim3, dim4))

    def  _process_yolo_output(self, outputs_reshaped, resolution_raw, conf_thres):
        """Take in a list of three reshaped YOLO outputs in (height,width,3,85) shape and return
        return a list of bounding boxes for detected object together with their category and their
        confidences in separate lists.
        
        Parameters
        ----------
        outputs_reshaped  : list
                            list of three reshaped YOLO outputs as NumPy arrays with shape (height,width,3,85)
        resolution_raw    : tuple
                            the original spatial resolution from the input PIL image in WH order
        conf_thres        : float
                            confidence threshold             
        """
        # E.g. in YOLOv3-608, there are three output tensors, which we associate with their
        # respective masks. Then we iterate through all output-mask pairs and generate candidates
        # for bounding boxes, their corresponding category predictions and their confidences:
        boxes, categories, confidences = list(), list(), list()
        for output, mask in zip(outputs_reshaped, self.masks):
            box, category, confidence = self._process_feats(output, mask)
            box, category, confidence = self._filter_boxes(box, category, confidence, conf_thres)
            boxes.append(box)
            categories.append(category)
            confidences.append(confidence)

        boxes = np.concatenate(boxes)
        categories = np.concatenate(categories)
        confidences = np.concatenate(confidences)

        # Scale boxes back to original image shape:
        width, height = resolution_raw
        image_dims = [width, height, width, height]
        boxes = boxes * image_dims

        # Using the candidates from the previous (loop) step, we apply the non-max suppression
        # algorithm that clusters adjacent bounding boxes to a single bounding box:
        nms_boxes, nms_categories, nscores = list(), list(), list()
        for category in set(categories):
            idxs = np.where(categories == category)
            box = boxes[idxs]
            category = categories[idxs]
            confidence = confidences[idxs]

            keep = self._nms_boxes(box, confidence)

            nms_boxes.append(box[keep])
            nms_categories.append(category[keep])
            nscores.append(confidence[keep])

        if not nms_categories and not nscores:
            return (np.empty((0, 4), dtype=np.float32),
                    np.empty((0, 1), dtype=np.float32),
                    np.empty((0, 1), dtype=np.float32))

        boxes = np.concatenate(nms_boxes)
        categories = np.concatenate(nms_categories)
        confidences = np.concatenate(nscores)

        return boxes, categories, confidences

    def _process_feats(self, output_reshaped, mask):
        """Take in a reshaped YOLOv4 output in height,width,3,85 format together with its
        corresponding YOLO mask and return the detected bounding boxes, the confidence,
        and the class probability in each cell/pixel.
        
        Parameters
        ----------
        output_reshaped : list
                          reshaped YOLOv4 output as NumPy arrays with shape (height,width,3,85)
        mask            : list of tuples
                          2-dimensional tuple with mask specification for this output 
        """
        def sigmoid_v(array):
            return np.reciprocal(np.exp(-array) + 1.0)

        def exponential_v(array):
            return np.exp(array)

        grid_h, grid_w, _, _ = output_reshaped.shape
        anchors = [self.anchors[i] for i in mask]

        # Reshape to N, height, width, num_anchors, box_params:
        anchors_tensor = np.reshape(anchors, [1, 1, len(anchors), 2])
        box_xy = sigmoid_v(output_reshaped[..., 0:2])
        box_wh = exponential_v(output_reshaped[..., 2:4]) * anchors_tensor
        box_confidence = sigmoid_v(output_reshaped[..., 4:5])
        box_class_probs = sigmoid_v(output_reshaped[..., 5:])

        col = np.tile(np.arange(0, grid_w), grid_h).reshape(-1, grid_w)
        row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_w)

        col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        grid = np.concatenate((col, row), axis=-1)

        box_xy += grid
        box_xy /= (grid_w, grid_h)
        box_wh /= self.input_wh
        box_xy -= (box_wh / 2.)
        boxes = np.concatenate((box_xy, box_wh), axis=-1)

        # boxes: centroids, box_confidence: confidence level, box_class_probs:
        # class confidence
        return boxes, box_confidence, box_class_probs

    
    def _filter_boxes(self, boxes, box_confidences, box_class_probs, conf_thres):
        """Take in the unfiltered bounding box descriptors and discard each cell
        whose score is lower than the object threshold set during class initialization.
        
        Parameters
        ----------
        boxes           : list
                          bounding box coordinates with shape (height,width,3,4); 4 for
                          x,y,height,width coordinates of the boxes
        box_confidences : list
                          bounding box confidences with shape (height,width,3,1); 1 for as
                          confidence scalar per element
        box_class_probs : list
                          class probabilities with shape (height,width,3,CATEGORY_NUM)
        conf_thes       : float
                          confidence threshold
        """
        box_scores = box_confidences * box_class_probs
        box_classes = np.argmax(box_scores, axis=-1)
        box_class_scores = np.max(box_scores, axis=-1)
        pos = np.where(box_class_scores >= self.conf_threshold)

        boxes = boxes[pos]
        classes = box_classes[pos]
        scores = box_class_scores[pos]

        return boxes, classes, scores

    def _nms_boxes(self, boxes, box_confidences):
        """Apply the Non-Maximum Suppression (NMS) algorithm on the bounding boxes with their
        confidence scores and return an array with the indexes of the bounding boxes we want to
        keep (and display later).
        
        Parameters
        ----------
        boxes           : numpy array
                          a NumPy array containing N bounding-box coordinates that survived filtering,
                          with shape (N,4); 4 for x,y,height,width coordinates of the boxes
        box_confidences : numpy array
                          box_confidences -- a Numpy array containing the corresponding confidences with shape N

        Returns
        -------
        keep            : 
        """

        x_coord = boxes[:, 0]
        y_coord = boxes[:, 1]
        width = boxes[:, 2]
        height = boxes[:, 3]

        areas = width * height
        ordered = box_confidences.argsort()[::-1]

        keep = []
        while ordered.size > 0:
            # Index of the current element:
            i = ordered[0]
            keep.append(i)
            xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
            yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
            xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
            yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

            width1 = np.maximum(0.0, xx2 - xx1 + 1)
            height1 = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = width1 * height1
            union = (areas[i] + areas[ordered[1:]] - intersection)

            # Compute the Intersection over Union (IoU) score:
            iou = intersection / union

            # The goal of the NMS algorithm is to reduce the number of adjacent bounding-box
            # candidates to a minimum. In this step, we keep only those elements whose overlap
            # with the current bounding box is lower than the threshold:
            indexes = np.where(iou <= self.nms_threshold)[0]
            ordered = ordered[indexes + 1]

        keep = np.array(keep)
        return keep

class EfficientDetPostprocess(object):
    """Class for post-processing two output tensors from EfficientDet-TensorRT model."""
    def __init__(self, conf_thres, nms_thres, input_size):
        """
        Initialize parameters, required for postprocessing efficientdet model outputs.

        Parameters
        ----------
        nms_thres   : int
                      Threshold value for performing non-maximum suppresion
        input_size  : int
                      input_size of the model
        """
        self.nms_thres  = nms_thres
        self.conf_thres = conf_thres
        self.input_size = input_size

    def _clip_boxes(self, predict_boxes, input_size=None):
        """
        Clip the invalid boxes such as
        1. negative values for width and height
        2. values greater than respective width and height

        Parameters
        ----------
        predict_boxes : numpy array
                        numpy array (num_of detection , 4) format
        input_size    : int
                        dimension of input image to the model
        """ 

        # use initialized value in postprocessing if no value is passed
        if input_size is None:
            input_size = self.input_size

        height, width = input_size, input_size
        predict_boxes[np.isnan(predict_boxes)] = 0
        predict_boxes[:, 0][predict_boxes[:, 0] < 0] = 0
        predict_boxes[:, 1][predict_boxes[:, 1] < 0] = 0

        predict_boxes[:, 2][predict_boxes[:, 2] > width]  = (width - 1) 
        predict_boxes[:, 3][predict_boxes[:, 3] > height] = (height - 1)

        return predict_boxes

    @staticmethod
    def _apply_nms(dets, scores, threshold):
        """
        aply non-maxumim suppression

        Parameters
        ----------
        dets      : numpy array
                    array in (num of dets x 4) format
        threshold : numpy array
                    array in (num of dets) format

        Retuens
        -------
        """
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1] # get boxes with more ious first

        keep = []
        while order.size > 0:
            i = order[0] # pick maxmum iou box
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1) # maximum width
            h = np.maximum(0.0, yy2 - yy1 + 1) # maxiumum height
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]

        return keep

    @staticmethod
    def _invert_affine(metas: Union[float, list, tuple], preds):
        if len(preds) == 0:
            return None
        else:
                new_w, new_h, old_w, old_h, padding_w, padding_h = metas
                preds[:, [0, 2]] = preds[:, [0, 2]] / (new_w / old_w)
                preds[:, [1, 3]] = preds[:, [1, 3]] / (new_h / old_h)
        return preds

    def __call__(self, boxes, classification, metas, cls=None, conf_thres=None, nms_thres=None):
        """
        Apply post-processing to model output
        """
        # use class initialized values for post-processing if no value is passed
        if conf_thres is None:
            conf_thres = self.conf_thres
        if nms_thres is None:
            nms_thres = self.nms_thres

        # clip the boxes
        boxes = self._clip_boxes(predict_boxes=boxes)

        # filter out detections with scores lower than confidence threshold
        scores = np.amax(classification, axis=1) 
        score_mask = scores > conf_thres
        scores = scores[score_mask]
        classification = np.argmax(classification[score_mask], axis=1)

        # filter out the boxes with mask
        filtered_boxes = boxes[score_mask]

        if cls is not None:
            cls_mask = classification == int(cls)
            filtered_boxes = filtered_boxes[cls_mask]
            scores = scores[cls_mask]
            
        keep = self._apply_nms(dets=filtered_boxes, scores=scores, threshold=nms_thres)
        calibrated_boxes = self._invert_affine(metas=metas, preds=filtered_boxes[keep])
        calibrated_scores = scores[keep]
        calibrated_classification = classification[keep]

        return calibrated_boxes, calibrated_scores, calibrated_classification

# TensorRT helper, getter, setter functions
class HostDeviceMem(object):
    """Simple helper data class that's a little nicer to use than a 2-tuple"""
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    """Allocates all host/device in/out buffers required for an engine."""

    inputs   = []
    outputs  = []
    bindings = []
    stream   = cuda.Stream()
    for binding in engine:
        size  = trt.volume(engine.get_binding_shape(binding)) * \
                engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        # Allocate host and device buffers.
        host_mem   = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes) 
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream):
    """do_inference (for TensorRT 7.0+)

    This function is generalized for multiple inputs/outputs for full
    dimension networks.
    Inputs and outputs are expected to be lists of HostDeviceMem objects.
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

class CascadeModel(object):
    """
    Cascade model is a class, which includes two TensorRT models processing in cascade manner.
    Cascade model : frame/data ==> YOLOv4 trt/EfficientDet trt model ==> detections => Feature Extractor trt model ==> features.
                               ==> digits trt model ==> timestamp

    The main reason why we need to create a class which combines execution of  two TensorRT is that
    1 : They operates in a cascade manner.
    2 : To be able to build/create multiple TensorRT builds and run them simultaneously,
    
    CUDA context needs to be created once and execution contexts need to be created for each TensorRT model.
    Multiple TensorRT execution contexts can use common CUDA context in a cascade manner.
    !!! Creating multiple CUDA contexts on a single GPU will cause nasty errors.

    WARNING!
    If we create a TensorRT inference class like this for every TensorRT model builds with
    own CUDA context, nasty CuskConvolution Error will happens.
    
    Cascade inferenceFurther Infos
    ------------------------------
    Issue#1 : https://forums.developer.nvidia.com/t/unable-to-run-two-tensorrt-models-in-a-cascade-manner
    Issue#2 : https://stackoverflow.com/questions/62719277/tensorrt-multiple-threads
    Issue#3 : https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-713/best-practices/index.html#thread-safety
    Issue#4 : https://github.com/NVIDIA/TensorRT/issues/301#issuecomment-570558499

    YOLOv4
    ------

    EfficientDet
    ------------
    EfficientDet model is a wrapper class for infernecing the EfficientDet-TensorRT model.
    EfficientDet Model has 7 variants by basing on the coefficient value.

    EfficientDet Model conversion process
    -------------------------------------
    EfficientDet Pytorch model ==> Freeze dynamic part in Implementation ==> Efficient TensorRT model

    EfficientDet Further Infos
    --------------------------
    EfficientDet : https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
    freeze dynamic parts : https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/issues/425

    Digits
    ------
    Simple ConvNet trained on digits dataset from cctv video data
    Recognize digits between 0 - 9
    """
    def _load_engine(self, engine_path):
        TRTbin = engine_path
        with open(TRTbin, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _create_context(self, engine):
        return engine.create_execution_context()

    def __init__(self, model, detect_engine_path, featext_engine_path, digits_engine_path,
                       nms_thres, conf_thres, num_classes=None):

        """
        Initialize parameters required for building Cascade model.

        Parameters
        ----------
        model               : str
                              Name of the Human Detection Model
                              Currently supports `YOLOv4` and `EfficientDet`
        detect_engine_path  : str
                              Path of the Human Detection TensorRT engine model file
        featext_engine_path : str
                              Path of the Feature Extractor TensorRT engine model file
        digits_engine_path  : str
                              Path of the Digits TensorRT engine model file
        nms_thres           : float(between 1 and 0)
                              Threshold value for performing non-maximum suppression
        conf_thres          : float (between 1 and 0)
                              Threshold value for filtering the boxes, outputted from model
        num_classs          : int
                              Total number of classes, that the model can detect

        Attributes
        ----------
        cuda_ctx            : CUDA context
        trt_logger          : TensorRT logger instance

        YOLOv4 Attributes
        -----------------
        yolo_masks          : list of tuples
                              A list of 3 three-dimensional tuples for the YOLO masks
        yolo_anchors        : list of tuples
                              A list of 9 two-dimensional tuples for the YOLO anchors
        yolo_output_shapes  : List of tuples (shape : (3, 3))
                              Desired output shapes of the YOLOv4 model
        yolo_masks          : list of tuples
        yolov4_input_shape  : tuple
                              input dimensions to YOLOv4 model
        postprocessor       : object
                              Collection of postprocessing functions such as nms, masking etc.

        EfficientDet Attributes
        -----------------------
        classes         : list
                          List of strings
        intput_sizes    : list
                          list of input sizes which the effficientdet-TensorRT model, accpects based on coefficient
        coefficient     : int
                          Compound coefficient value for scaling EfficientDet model scaling
        postprocessor   : object
                          Collection of postprocessing functions such as nms, clipboxes etc.
        """
        # create a common CUDA context, to be used by two tensorRT engines
        self.cuda_ctx = cuda.Device(0).make_context() # use GPU:0

        self.supported_models = ['yolov4', 'efficientdet']

        # check if model is in supported models list or not
        assert model.lower().replace(" ", "") in self.supported_models, "Mentioned model : `{}` is not supported!".format(model)
        self.model = model.lower().replace(" ", "")

        self.detect_engine_path = str(detect_engine_path)
        self.featext_engine_path = str(featext_engine_path)
        self.digits_engine_path = str(digits_engine_path)

        # check if the engine files exists
        assert ops.isfile(self.detect_engine_path), "Human Detection Engine file does not exists. Please check the path!"
        assert ops.isfile(self.featext_engine_path), "Feature Extractor Engine file does not exists. Please check the path!"
        assert ops.isfile(self.digits_engine_path), "Digits Engine file does not exists. Please check the path!"

        print("Specified model : `{}` is supported!".format(self.model))
        print("Using `{}` as human detection model...".format(self.model))

        # threshold values
        self.nms_thres  = nms_thres
        self.conf_thres = conf_thres

        # Initialize Parameters required for building detector model
        if self.model == 'yolov4':
            self._initialize_yolov4_parameters()
        elif self.model == 'efficientdet':
            self._initialize_efficientdet_parameters()

        # setup inference function
        self.inference_fn = do_inference

        # setup logger
        self.trt_logger = trt.Logger(trt.Logger.INFO)

        # load engines
        self.detect_engine  = self._load_engine(self.detect_engine_path)
        self.featext_engine = self._load_engine(self.featext_engine_path)
        self.digits_engine = self._load_engine(self.digits_engine_path)

        # initialize pre-process function for digits classifier TensorRT model
        self.digits_preprocess = DigitsPreprocess(width=28, height=28)

        # setup parameters for feature extractor body
        self.featext_input_shape = (64, 128)
        self.mean                = [0.485, 0.456, 0.406]
        self.std                 = [0.229, 0.224, 0.225]

        try:
            self.detect_context  = self._create_context(self.detect_engine)
            self.featext_context = self._create_context(self.featext_engine)
            self.digits_context  = self._create_context(self.digits_engine)

            self.detect_inputs, self.detect_outputs, self.detect_bindings, self.detect_stream = \
                allocate_buffers(self.detect_engine)

            self.featext_inputs, self.featext_outputs, self.featext_bindings, self.featext_stream = \
                allocate_buffers(self.featext_engine)
            
            self.digits_inputs, self.digits_outputs, self.digits_bindings, self.digits_stream = \
                allocate_buffers(self.digits_engine)
        except Exception as e:
            self.cuda_ctx.pop()
            del self.cuda_ctx
            raise RuntimeError("Fail to allocate CUDA resources") from e
        
    def __del__(self):
        """Free CUDA memories"""
        del self.detect_stream
        del self.detect_outputs
        del self.detect_inputs

        del self.featext_stream
        del self.featext_outputs
        del self.featext_inputs

        del self.digits_stream
        del self.digits_outputs
        del self.digits_inputs

        self.cuda_ctx.pop()
        del self.cuda_ctx

    def _initialize_yolov4_parameters(self):
        """
        Initialize Parameters required for building YOLOv4 TensorRT runtime model.
        """
        self.input_size = int(self.detect_engine_path.split("-")[-1].split('.')[0])

        self.yolov4_input_shape = (self.input_size, self.input_size)

        # build preprocessor instance
        self.preprocess = YOLOPreprocess()

        # YOLOv4 can detect 80 classes
        # setup yolov4 post processing parameters
        self.num_classes = 80
        filters = (self.num_classes + 5) * 3
        h, w    = self.yolov4_input_shape
        self.yolo_masks           = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
        self.yolov4_output_shapes = [(1, filters, h //  8, w //  8),
                                     (1, filters, h // 16, w // 16),  
                                     (1, filters, h // 32, w // 32)]
        self.yolo_anchors         = [(12, 16), (19, 36), (40, 28),
                                     (36, 75), (76, 55), (72, 146),
                                     (142, 110), (192, 243), (459, 401)]

        # setup postprocesser for YOLOv4
        self.postprocessor = YOLOv4Postprocess(yolo_masks=self.yolo_masks, yolo_anchors=self.yolo_anchors, 
                                               conf_threshold=self.conf_thres,nms_threshold=self.nms_thres,
                                               yolo_input_resolution=self.yolov4_input_shape, num_classes=self.num_classes)
    
    def _initialize_efficientdet_parameters(self):
        """
        Initialize Parameters required for building EfficientDet TensorRT runtime model.
        """
        # input_sizes
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        # output sizes
        self.output_sizes = [49104, 76725, 110484, 150381, 196416, 306900, 306900, 441936]
        # classes
        self.num_classes = 90
        self.classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                        'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                        'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
                        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                        'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                        'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                        'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
                        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                        'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                        'toothbrush']

        self.coefficient = int(list(self.detect_engine_path.split(".")[0])[-1])

        # preprocessing parameters
        self.mean=(0.406, 0.456, 0.485)
        self.std=(0.225, 0.224, 0.229)
        self.input_size = int(self.input_sizes[self.coefficient])

        # build preprocessor instance
        self.preprocess = EfficientDetPreprocess(mean=self.mean, std=self.std)

        # postprocessing parameters
        self.output_size = self.output_sizes[self.coefficient]
        # self.output_size = 500
        self.postprocess = EfficientDetPostprocess(conf_thres=self.conf_thres, nms_thres=self.nms_thres,
                                                   input_size=self.input_size)
                                        
    def extract(self, img, boxes):
        "Extract features from detected objects"
        
        if len(boxes) !=  0:
            # convert to rgb image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # build a container for the outputs
            outputs = np.zeros([len(boxes), 512], dtype=np.float32)

            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box.astype(np.int)
                img_crop = img[y1 : y2, x1 : x2]
                preprocessed_img_crop = _preprocess_fe(img_crop, self.featext_input_shape, 
                                                    self.mean, self.std)

                # Set host input to the image. The do_inference() function
                # will copy the input to the GPU before executing
                self.featext_inputs[0].host =  np.ascontiguousarray(preprocessed_img_crop)
                trt_features = self.inference_fn(
                    context  = self.featext_context,
                    bindings = self.featext_bindings,
                    inputs   = self.featext_inputs,
                    outputs  = self.featext_outputs,
                    stream   = self.featext_stream
                )

                outputs[idx] = np.array(trt_features)

            return outputs
        else:
            return np.array([])

    def detect(self, img, cls=None):
        """
        Detect objects in the input image with selected detection model.

        Parameters
        ----------
        img  : numpy array
               Image numpy array, read with cv2
        """

        assert cls < self.num_classes and cls >= 0, "filter class doesn't exists!"

        if self.model == 'yolov4':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            shape_orig_WH = (img.shape[1], img.shape[0])

            # preprocess the image

            # Set host input to the image. The do_inference() function
            # will copy the input to the GPU before executing.
            img_reized = self.preprocess(img, (self.input_size, self.input_size))

            self.detect_inputs[0].host = np.ascontiguousarray(img_reized)

            trt_outputs  = self.inference_fn(
                context  = self.detect_context,
                bindings = self.detect_bindings,
                inputs   = self.detect_inputs,
                outputs  = self.detect_outputs,
                stream   = self.detect_stream
            )

            # Before doing post-processing, we need to reshape the outputs
            # as do_inference() will give us flat arrays.
            trt_outputs = [
                output.reshape(shape)
                for output, shape in zip(trt_outputs, self.yolov4_output_shapes)
            ]
            # Run the post-processing algorithms on the TensorRT outputs
            # and get the bounding box details of detected objects
            boxes, classes, scores = self.postprocessor.process(
                trt_outputs, shape_orig_WH, self.conf_thres)

            if boxes is None:
                return np.empty((0, 4), dtype=np.float32), np.array([]), np.array([])
            else:
                cls_mask = classes == int(cls)
                boxes   = boxes[cls_mask]
                scores  = scores[cls_mask]
                classes = classes[cls_mask]
                return boxes, scores, classes

        elif self.model == "efficientdet":
            img_reized, metas = self.preprocess(img, (self.input_size, self.input_size))

            self.detect_inputs[0].host = np.ascontiguousarray(img_reized)

            trt_outputs  = self.inference_fn(
                context  = self.detect_context,
                bindings = self.detect_bindings,
                inputs   = self.detect_inputs,
                outputs  = self.detect_outputs,
                stream   = self.detect_stream
            )

            classification = np.reshape(trt_outputs[0], (self.output_size, self.num_classes))
            boxes          = np.reshape(trt_outputs[1], (self.output_size, 4))

            pred_boxes, pred_scores, pred_classes = self.postprocess(boxes=boxes, classification=classification,
                                                                    metas=metas, cls=cls)

            if pred_boxes is None:
                return np.empty((0, 4), dtype=np.float32), np.array([]), np.array([])
            else:
                return pred_boxes, pred_scores, pred_classes

    def extract_timestamp(self, img):
        """
        Extract digits from given cctv image frame.

        Parameters
        ----------
        img : numpy array
              Image numpy array in BGR format

        Returns
        -------
        timestamp : str
                    Timestamp of current given frame
        """
        batch_preprocessed_imgs = self.digits_preprocess(image=img)

        # Set host input to the image. The do_inference() function
        # will copy the input to the GPU before executing
        self.digits_inputs[0].host = np.ascontiguousarray(batch_preprocessed_imgs)
        trt_outputs = self.inference_fn(
            context=self.digits_context,
            bindings=self.digits_bindings,
            inputs=self.digits_inputs,
            outputs=self.digits_outputs,
            stream=self.digits_stream
        )

        outputs = np.reshape(trt_outputs[0], (14, 10))
        preds = np.argmax(outputs, axis=1)

        # Date in mm-dd-yyyy
        date = str(preds[0]) + str(preds[1]) + "-" + str(preds[2]) + str(preds[3]) + "-" + str(preds[4]) + str(preds[5]) + str(preds[6]) + str(preds[7])
        # Time in hh:mm:ss
        time = str(preds[8]) + str(preds[9]) + ":" + str(preds[10]) + str(preds[11]) + ":" + str(preds[12]) + str(preds[13])

        return date, time

    def infer(self, img, cls=0):
        """
        Perform cascade inferencing with Cascade model.
        """
        # Perform human detection
        boxes, scores, classes = self.detect(img=img, cls=cls)
        date, time = self.extract_timestamp(img=img)

        # extract features from detected boxes
        features = self.extract(img, boxes)

        preds = {}
        # construct a new dict for storing the cascaded inference results
        if len(boxes) == 0:
            preds['rois']     = np.zeros((0, 4))
            preds['scores']   = scores
            preds['classes']  = classes
            preds['features'] = features
            preds['date']     = date
            preds['time']     = time
        else:
            preds['rois']     = np.asarray(boxes, dtype=np.float32) if not isinstance(boxes, np.ndarray) else boxes
            preds['scores']   = np.asarray(scores, dtype=np.float32) if not isinstance(scores, np.ndarray) else scores
            preds['classes']  = np.asarray(classes, dtype=np.float32) if not isinstance(classes, np.ndarray) else classes
            preds['features'] = np.asarray(features, dtype=np.float32)
            preds['date']     = date
            preds['time']     = time

        return preds


