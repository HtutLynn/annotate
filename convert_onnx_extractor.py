import sys
import os

import numpy as np
import cv2
import torch
import onnx
import onnxruntime

# import model\
from extractor.model import Net

def transform_to_onnx(weight_file, batch_size, IN_IMAGE_H, IN_IMAGE_W):
    """
    Converts pytorch weights file into onnx runtime model
    
    Parameters
    ----------
    weight_file : str
                  path of the pytorch weight file
    batch_size  : int
                  static batchsize of the onnx model
    IN_IMAGE_H  : int
                  static height of the input image
    IN_IMAGE_W  : int
                  static width of the input image   
    """

    model = Net()
    pretrained_dict = torch.load(weight_file)
    model.load_state_dict(pretrained_dict)
    model.cuda()

    # construct dummy data with static batch size
    x = torch.randn((batch_size, 3, IN_IMAGE_H, IN_IMAGE_W), requires_grad=True).cuda()

    # file name
    onnx_file_name = "extractor_{}.onnx".format(batch_size)

    # Export the onnx model
    print('Export the onnx model....')
    torch.onnx.export(model,
                      x,
                      onnx_file_name,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes=None)
    
    print("Onnx model exporting done!")
    return onnx_file_name

def extract(session, image_src):
    """Extract the features of the image using onnx model

    Parameters
    ----------
    session   : onnx runtime session instance
    
    image_src : numpy uint8 array
                image file, read by cv2

    Return
    ------
    features  : numpy array
                features, inferred from onnxruntime session
    """

    # reshape the image
    img  = cv2.resize(image_src, (64, 128))

    # swap dimensions for rgb format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # transpose the image to pytorch format since the model is converted from pytorch model
    img =  np.transpose(img, (2, 0, 1)).astype(np.float32)

    # unsqueeze extra dimension
    img = np.expand_dims(img, axis=0)

    # squeeze the valueof each pixel to 0 - 1 range
    img /= 255.0

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    # perform normalization per channel
    img[:, 0, :, :] = img[:, 0, :, :] - mean[0] / std[0]
    img[:, 1, :, :] = img[:, 1, :, :] - mean[1] / std[1]
    img[:, 2, :, :] = img[:, 2, :, :] - mean[2] / std[2]

    input_name = session.get_inputs()[0].name 
    outputs =session.run(None, {input_name : img})
    features = np.array(outputs)

    return features

def main(weight_file, image_path, batch_size, IN_IMAGE_H, IN_IMAGE_W):

    # Transform the onnx as specified batch size
    onnx_file_name = transform_to_onnx(weight_file, batch_size, IN_IMAGE_H, IN_IMAGE_W)

    # do test inferencing with converted onnx model
    session = onnxruntime.InferenceSession(onnx_file_name)
    print(session)

    # do inference
    image_src =cv2.imread(image_path)
    features = extract(session, image_src)
    print("shape of features : {}".format(features.shape))

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


# python convert_onnx.py 'extractor/extractor.pt' 'dogcrop.png' 1 128 64
# python convert_onnx_extractor.py 'checkpoints/extractor.pt' 'dogcrop.jpg' 1 128 64