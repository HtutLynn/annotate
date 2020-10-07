from __future__ import print_function

import numpy as np
import cv2
import time
import os.path as ops
import tensorrt as trt
import pycuda.driver as cuda

def _preprocess_frame(img, input_shape):
    """Preprocess an image before TRT YOLO inferencing.

    Parameters
    ----------
    img         : numpy array
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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img /= 255.0
    return img

class Postprocess(object):
    """Class for post-processing the three output tensors from YOLO."""

    def __init__(self, yolo_masks, yolo_anchors,conf_threshold, 
                 nms_threshold, yolo_input_resolution, num_classes):
        """Initialize with all values that will be kept when processing
        several frames.  Assuming 3 outputs of the network in the case
        of (large) YOLO, or 2 for the Tiny YOLO.

        Parameters
        ----------
        yolo_masks          : list
                              a list of 3 (or 2) three-dimensional tuples for the YOLO masks
        yolo_anchors        : list 
                              a list of 9 (or 6) two-dimensional tuples for the YOLO anchors
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
        """Take the YOLO outputs generated from a TensorRT forward pass, post-process them
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
        """Take in a reshaped YOLO output in height,width,3,85 format together with its
        corresponding YOLO mask and return the detected bounding boxes, the confidence,
        and the class probability in each cell/pixel.
        
        Parameters
        ----------
        output_reshaped : list
                          reshaped YOLO output as NumPy arrays with shape (height,width,3,85)
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

        keep = list()
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

class TrtYOLOv4(object):
    """TrtYOLOv4 class encapsulates things needed to run TRT YOLO.
    """

    def _load_engine(self):
        TRTbin = self.engine_path
        with open(TRTbin, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _create_context(self):
        return self.engine.create_execution_context()

    def __init__(self, engine_path, input_shape, 
                nms_thres, conf_thres, num_classes=80):

        """Initialize parameters requried for building TensorRT engine.
           TensorRT plugins, engine and context.

        Parameters
        ----------
        engine_path : str
                      Path of the TensorRT engine model file
        input_shape : tuple
                      a tuple of (H, W)
        nms_thres   : float(between 1 and 0)
                      Threshold value for performing non-maximum suppression
        conf_thres  : float (between 1 and 0)
                      Threshold value for filtering the boxes, outputted from model
        num_classs  : int
                      Total number of classes, that the model can detect
        yolo_masks  : list of tuples
                      A list of 3 three-dimensional tuples for the YOLO masks
        yolo_anchors: list of tuples
                      A list of 9 two-dimensional tuples for the YOLO anchors

        Returns
        -------
        TensorRT engine instance capable of inferencing on images
        """
        self.cuda_ctx    = cuda.Device(0).make_context() # Use GPU:0

        self.engine_path = engine_path
        # check whether the file exists or not
        assert ops.exists(self.engine_path), "Engine file does not exist. Please check!"

        self.input_shape = input_shape
        self.nms_thres   = nms_thres
        self.conf_thres  = conf_thres
        self.num_classes = num_classes

        # Setup YOLOv4 postprocessing parameters
        filters = (self.num_classes + 5) * 3
        h, w = self.input_shape
        self.yolo_masks   = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
        self.output_shapes = [(1, filters, h //  8, w //  8),
                             (1, filters, h // 16, w // 16),  
                             (1, filters, h // 32, w // 32)]
        self.yolo_anchors = [(12, 16), (19, 36), (40, 28),
                             (36, 75), (76, 55), (72, 146),
                             (142, 110), (192, 243), (459, 401)]
        self.yolo_input_resolution = self.input_shape 

        # setup inference function
        self.inference_fn = do_inference

        # setup logger
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.engine     = self._load_engine()

        # setup postprocess
        self.postprocessor = Postprocess(yolo_masks=self.yolo_masks, yolo_anchors=self.yolo_anchors, conf_threshold=self.conf_thres,
                                         nms_threshold=self.nms_thres, yolo_input_resolution=self.input_shape, num_classes=self.num_classes)

        try:
            self.context = self._create_context()
            self.inputs, self.outputs, self.bindings, self.stream = \
                allocate_buffers(self.engine)
        except Exception as e:
            self.cuda_ctx.pop()
            del self.cuda_ctx
            raise RuntimeError("Fail to allocate CUDA resources") from e

    def __del__(self):
        """Free CUDA memories"""
        del self.stream
        del self.outputs
        del self.inputs
        self.cuda_ctx.pop()
        del self.cuda_ctx
    
    def detect(self, img):
        """Detect objects in the input image."""
        shape_orig_WH = (img.shape[1], img.shape[0])
        t0 = time.time()
        img_resized = _preprocess_frame(img, self.input_shape)

        # Set host input to the image. The do_inference() function
        # will copy the input to the GPU before executing.
        t1 = time.time()
        self.inputs[0].host = np.ascontiguousarray(img_resized)
        trt_outputs = self.inference_fn(
            context=self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream)
        t2 = time.time()

        # Before doing post-processing, we need to reshape the outputs
        # as do_inference() will give us flat arrays.
        trt_outputs = [
            output.reshape(shape)
            for output, shape in zip(trt_outputs, self.output_shapes)]
        # Run the post-processing algorithms on the TensorRT outputs
        # and get the bounding box details of detected objects
        boxes, classes, scores = self.postprocessor.process(
            trt_outputs, shape_orig_WH, self.conf_thres)
        t3 = time.time()

        print("Preprocess time      : {}".format(t1 - t0))
        print("Model inference time : {}".format(t2 - t1))
        print("Postprocess time     : {}".format(t3 - t2))
        return boxes, scores, classes