import os
import subprocess

# import azure storage blob python inferface
from utils.storage import BlobClient

dest = "checkpoints"

# Create blob client instance
client = BlobClient(connection_string=os.environ['AZURE_STORAGE_CONNECTION_STRING'],
                    container_name='Downloadblob')

# Download pytorch and ONNX model files for EfficientDet
client.download(source="EfficientDet", dest=dest)

# Download pytorch and ONNX model files for DeepSORT feature Extractor
client.download(source="Extractor", dest=dest)

# Download pytorch and ONNX model files for YOLOv4
client.download(source="YOLOv4", dest=dest)
