from __future__ import print_function

import cv2
import time
import numpy as np
import os.path as ops
import tensorrt as trt
from typing import Union

import pycuda.driver as cuda

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

    # resize the image by maintaining aspect ratio
    if new_w != old_w or new_h != old_h:
        if interpolation is None:
            image = cv2.resize(image, (new_w, new_h))
        else:
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    padding_h = height - new_h
    padding_w = width - new_w

    # pad the resized-contrast ratio maintained image to get desired dimensions
    image = cv2.copyMakeBorder(image, 0, 0, padding_h, padding_w, cv2.BORDER_CONSTANT, value=means)

    return image, new_w, new_h, old_w, old_h, padding_w, padding_h,

def _preprocess(img, input_shape, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
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

    normalized_img = (img / 255 - mean) / std
    # This is where the freaking errors lie
    # I kinda normalized the input images but didn't pass the normalized_img to padding function
    # therefore, pixels distribution is a mess so that's why model was generating garbage values
    img_meta = _aspectaware_resize_padding(image=normalized_img, width=input_shape, height=input_shape,
                                            interpolation=None, means=None)
    img = img_meta[0].transpose((2, 0, 1)).astype(np.float32)
    # img = img[np.newaxis, ...].astype(np.float32)

    return img, img_meta[1:]

class postprocess(object):
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

class EfficientDetModel(object):
    """
    EfficientDet model is a wrapper class for infernecing the EfficientDet-TensorRT model.
    EfficientDet Model has 7 variants by basing on the coefficient value.

    EfficientDet Model conversion process
    -------------------------------------
    EfficientDet Pytorch model ==> Freeze dynamic part in Implementation ==> Efficient TensorRT model

    Further Infos
    -------------
    EfficientDet : https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
    freeze dynamic parts : https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/issues/425
    """

    def _load_engine(self, engine_path):
        TRTbin = engine_path
        with open(TRTbin, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _create_context(self, engine):
        return engine.create_execution_context()

    def __init__(self, engine_path, nms_thres, conf_thres, num_classes=90):
        """
        Initialize the parameters required for building EfficientDet-TensorRT model.

        Parameters
        ----------
        engine_path : str
                      Path of the EfficientDet TensorRT engine model file
        nms_thres   : int
                      Threshold value for performing non-maximum suppresion
        conf_thres  : int
                      Threshold value for filtering the boxes, outputted from the model
        num_classes : int
                      Total number of classes, that the model can detect

        Attributes
        ----------
        classes         : list
                          List of strings
        intput_sizes    : list
                          list of input sizes which the effficientdet-TensorRT model, accpects based on coefficient
        coefficient     : int
                          Compound coefficient value for scaling EfficientDet model scaling
        trt_logger      : TensorRT Logger instance
        cuda_ctx        : CUDA context
        postprocessor   : object
                          Collection of postprocessing functions such as nms, clipboxes etc.
        """

        # create a CUDA context, to be used by TensorRT engine
        self.cuda_ctx = cuda.Device(0).make_context() # use GPU:0

        self.engine_path = str(engine_path)

        # threshold values
        self.nms_thres  = nms_thres
        self.conf_thres = conf_thres

        # input_sizes
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        # output sizes
        self.output_sizes = [49104, 76725, 110484, 150381, 196416, 306900, 306900, 441936]
        # classes
        self.num_classes = num_classes
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

        # check if the engine file exists
        assert ops.isfile(self.engine_path), "EfficientDet Engine file does not exists. please check the path!"

        # extract coefficient value from path
        print(self.engine_path)
        self.coefficient = int(list(self.engine_path.split(".")[0])[-1])


        # preprocessing parameters
        self.mean=(0.406, 0.456, 0.485)
        self.std=(0.225, 0.224, 0.229)
        self.input_size = int(self.input_sizes[self.coefficient])

        # postprocessing parameters
        self.output_size = self.output_sizes[self.coefficient]
        # self.output_size = 500
        self.postprocess = postprocess(conf_thres=self.conf_thres, nms_thres=self.nms_thres, input_size=self.input_size)

        # make inference function instance
        self.inference_fn = do_inference
        # setup logger
        self.trt_logger = trt.Logger(trt.Logger.INFO)

        # load engine
        self.engine = self._load_engine(self.engine_path)

        try:
            self.context = self._create_context(self.engine)
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
        
        # release the memory occupied by cuda context creation
        self.cuda_ctx.pop()
        del self.cuda_ctx

    def detect(self, img, cls=None):
        """
        Detect people in the input image.
        Perform inferencing with EfficientDet-TensorRT model
        
        Parameters
        ----------
        img  : numpy array
               Image numpy array, read with cv2
        """


        assert cls < len(self.classes) and cls >= 0, "filter class doesn't exists!"

        t0 = time.time()
        preprocessed_img, metas  = _preprocess(img=img, input_shape=self.input_size,
                                               mean=self.mean, std=self.std)
        t1 = time.time()
        
        # Set host input to the image. The do_inference() function
        # will copy the input to the GPU before executing.
        self.inputs[0].host = np.ascontiguousarray(preprocessed_img)
        trt_outputs = self.inference_fn(
            context=self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream
            )
        t2 = time.time()

        classification = np.reshape(trt_outputs[0], (self.output_size, self.num_classes))
        boxes          = np.reshape(trt_outputs[1], (self.output_size, 4))

        pred_boxes, pred_scores, pred_classes = self.postprocess(boxes=boxes, classification=classification,
                                                                 metas=metas, cls=cls)
        t3 = time.time()

        print("Preprocessing  : {}".format(t1 - t0))
        print("Inference      : {}".format(t2 - t1))
        print("postprocessing : {}".format(t3 - t2))
        if pred_boxes is None:
            return np.empty((0, 4), dtype=np.float32), np.array([]), np.array([])
        else:
            return pred_boxes, pred_scores, pred_classes