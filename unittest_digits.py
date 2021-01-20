from __future__ import print_function

import cv2
import time
import numpy as np
import os.path
import tensorrt as trt

import pycuda.autoinit
import pycuda.driver as cuda

def _preprocess(image, width, height, interpolation=None):
    """
    Preprocess an image for digits classifier TensorRT model inference

    Parameters
    ----------
    image  : numpy array
             gray-scale image in (1, height, width) format
    width  : int
             width of the result pre-processed image
    height : int
             height of the result pre-processed image
    interpolation : cv2 resize interpolation method
                    method, used by cv2 for resize         
    """
    resized_image = cv2.resize(image, (height, width))
    preprocessed_image = (resized_image/255.0 - 0.1307) / 0.3081
    preprocessed_image = preprocessed_image.transpose((2, 0, 1)).astype(np.float32)
    
    return preprocessed_image

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

class DigitRecognizer(object):
    """
    DigitRecognizer model is a wrapper class for inferencing the digits TensorRT model.

    DigitRecognizer Model conversion process
    ----------------------------------------
    Digits Recognizer Pytorch ==> Onnx Model ==> TensorRT runtime model

    Further Infos
    -------------
    Simple ConvNet for recognizing Digits in CCTV cameras videos.
    Requirs precise crop of digits from camera frame.
    """
    def _load_engine(self, engine_path):
        TRTbin = engine_path
        with open(TRTbin, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _create_context(self, engine):
        return engine.create_execution_context()

    def __init__(self, engine_path, input_size, num_classes=10):
        """
        Initialize the parameters, required for building DigitsRecognizer TensorRT model.
        
        Parameters
        ----------
        engine_path : str
                      Path of the DigitsRecognizer-TensorRT engine model file
        input_size  : List
                      Dimensions of input image to TensorRT model : [height, width]
        num_classes : int
                      Number of classes, that model can detect

        Attributes
        ----------
        trt_logger  : TensorRT Logger instance
        cuda_ctx    : CUDA context
        """

        # create a CUDA context, to be used by TensorRT engine
        self.cuda_ctx = cuda.Device(0).make_context() # use GPU:0
        self.engine_path = str(engine_path)

        self.input_size = input_size

        # classes
        self.num_classes = num_classes

        # check if the engine file exists or not
        assert os.path.isfile(self.engine_path), "DigitsRecognizer TensorRT file does not exists."

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

    def classify(self, img):
        """
        Classify digits by precisely croping digits image from the given frame.

        img : numpy array
              Image array in BGR format
        """
        assert isinstance(img, np.ndarray), "Given image is not numpy array!"

        batch_img = np.zeros((14, 3, self.input_size[0], self.input_size[1]), dtype=np.float32)

        batch_img[0] = _preprocess(img[56:97, 43:72].copy(), width=self.input_size[1], height=self.input_size[0])
        batch_img[1] = _preprocess(img[56:97, 71:95].copy(), width=self.input_size[1], height=self.input_size[0])
        batch_img[2] = _preprocess(img[56:97, 120:143].copy(), width=self.input_size[1], height=self.input_size[0])
        batch_img[3] = _preprocess(img[56:97, 141:167].copy(), width=self.input_size[1], height=self.input_size[0])
        # cv2.imwrite("Day2.jpg", grayscale_img[56:97, 141:167].copy())
        batch_img[4] = _preprocess(img[56:97, 191:215].copy(), width=self.input_size[1], height=self.input_size[0])
        batch_img[5] = _preprocess(img[56:97, 214:240].copy(), width=self.input_size[1], height=self.input_size[0])
        batch_img[6] = _preprocess(img[56:97, 238:264].copy(), width=self.input_size[1], height=self.input_size[0])
        batch_img[7] = _preprocess(img[56:97, 262:289].copy(), width=self.input_size[1], height=self.input_size[0])
        batch_img[8] = _preprocess(img[56:97, 405:434].copy(), width=self.input_size[1], height=self.input_size[0])
        batch_img[9] = _preprocess(img[56:97, 430:458].copy(), width=self.input_size[1], height=self.input_size[0])
        batch_img[10] = _preprocess(img[56:97, 476:504].copy(), width=self.input_size[1], height=self.input_size[0])
        batch_img[11] = _preprocess(img[56:97, 502:529].copy(), width=self.input_size[1], height=self.input_size[0])
        batch_img[12] = _preprocess(img[56:97, 548:576].copy(), width=self.input_size[1], height=self.input_size[0])
        batch_img[13] = _preprocess(img[56:97, 573:603].copy(), width=self.input_size[1], height=self.input_size[0])
        # cv2.imwrite("last.jpg", grayscale_img[56:97, 573:603].copy())

        # Set host input to the image. The do_inference() function
        # will copy the input to the GPU before executing.
        self.inputs[0].host = np.ascontiguousarray(batch_img)
        trt_outputs = self.inference_fn(
            context=self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream
        )

        outputs = np.reshape(trt_outputs[0], (14, 10))
        preds = np.argmax(outputs, axis=1)

        # Date in mm-dd-yyyy
        date = str(preds[0]) + str(preds[1]) + "-" + str(preds[2]) + str(preds[3]) + "-" + str(preds[4]) + str(preds[5]) + str(preds[6]) + str(preds[7])
        # Time in hh:mm:ss
        time = str(preds[8]) + str(preds[9]) + ":" + str(preds[10]) + str(preds[11]) + ":" + str(preds[12]) + str(preds[13]) 

        return date, time

if __name__ == "__main__":
    image = cv2.imread("sample/digits.png")
    model = DigitRecognizer(engine_path="checkpoints/trt/digits.trt", input_size=(28, 28), num_classes=10)


    f_date, f_time = model.classify(img=image)
    print(f_date)
    print(f_time)

    t0 = time.time()
    for _ in range(30):
        _, _ = model.classify(image)

    t1 = time.time()

    print("one frame inference time : {}".format((t1 - t0)/ 30))
    FPS = 1/((t1 - t0) / 30)
    print("FPS : {}".format(FPS))