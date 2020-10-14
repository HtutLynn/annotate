from os import pread
from numpy.lib.shape_base import expand_dims
import onnx
import onnxruntime
import torch
import cv2
import sys
import numpy as np

# import custom libraries
from digits.preprocessing import preprocess
from model import Net

def transform_to_onnx(weight_file, batch_size, IN_IMAGE_H, IN_IMAGE_W):
    """
    Converts pytorch weights file into onnx runtime model

    Parameters
    ----------
    weight_file : str
                  path of the Pytorch weight file
    batch_size  : int
                  static batch size of the onnx model
    IN_IMAGE_H  : int
                  static height of the input image
    IN_IMAGE_W  : int
                  static width of the input image
    """

    model = Net()
    model = torch.load(weight_file)
    model.cuda()

    # construct dummy data with static batch size
    x = torch.randn((batch_size, 1, IN_IMAGE_H, IN_IMAGE_W), requires_grad=True).cuda()

    # file name
    onnx_file_name = "digits_{}.onnx".format(batch_size)

    # Export the onnx model
    print("Exporting the onnx model...")
    torch.onnx.export(model,
                      x,
                      onnx_file_name,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes=None)

    print('Onnx model exporting done!')
    return onnx_file_name

def main(weight_file, image_path, batch_size, IN_IMAGE_H, IN_IMAGE_W):
    """
    """
    # Transform the onnx as specified batch size
    onnx_file_name = transform_to_onnx(weight_file, batch_size, IN_IMAGE_W, IN_IMAGE_H)

    # do test inference
    session = onnxruntime.InferenceSession(onnx_file_name)
    date_time_list = preprocess(image_path)
    batch_image = np.zeros((len(date_time_list), 1 , IN_IMAGE_H, IN_IMAGE_W))
    for idx, img in enumerate(date_time_list):
        batch_image[idx] = date_time_list[idx].astype(np.float32)

    print(batch_image.shape)

    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name : batch_image.astype(np.float32)})
    result = result[0]
    print(result.shape)

    # Date in mm-dd-yyyy
    date = str(result[0]) + str(result[1]) + "-" + str(result[2]) + str(result[3]) + "-" + str(result[4]) + str(result[5]) + str(result[6]) + str(result[7])
    
    # Time in hh:mm:ss
    time = str(result[8]) + str(result[9]) + ":" + str(result[10]) + str(result[11]) + ":" + str(result[12]) + str(result[13]) 

if __name__ == "__main__":
    print("Converting to onnx and running demo....")

    if len(sys.argv) == 6:

        weight_file = sys.argv[1]
        image_path  = sys.argv[2]
        batch_size  = int(sys.argv[3])
        IN_IMAGE_H  = int(sys.argv[4])
        IN_IMAGE_W  = int(sys.argv[5])

        main(weight_file, image_path, batch_size, IN_IMAGE_H, IN_IMAGE_W)

    else:
        print("python convert_onnx.py <weight_file> <image_path> <batch_size> <IN_IMAGE_H> <IN_IMAGE_W>")
        # python convert_onnx_digits.py checkpoints/my_model_KLDiv_linear.pth digits.png 14 28 28