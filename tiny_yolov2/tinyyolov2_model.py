#!/usr/bin/env python

# MIT License
#
# Copyright (c) 2021 MACNICA-CLAVIS-NV
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
import wget
import tarfile
import shutil


MODEL_URL = 'https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.tar.gz'
LABEL_URL = 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/voc.names'
LABEL_FILE = os.path.basename(LABEL_URL)
MODEL_FILE = 'model.onnx'
CONFIG_FILE = 'config.pbtxt'
CONFIG_DATA = '''name: "tinyyolov2_onnx"
platform: "onnxruntime_onnx"
max_batch_size: 128
input [
    {
        name: "image"
        data_type: TYPE_FP32
        format: FORMAT_NCHW
        dims: [3, 416, 416]
    }
]
output [
    {
        name: "grid"
        data_type: TYPE_FP32
        dims: [125, 13, 13]
        label_filename: "voc.names"
    }
]
'''


def download_file(url, path):
    file = os.path.join(path, os.path.basename(url))
    if not os.path.exists(file):
        wget.download(url, out=file)
    else:
        print('{} already exists'.format(file))
    return file

def extract_onnx_file(path):
    model_file = os.path.join(path, MODEL_FILE)
    if not os.path.exists(model_file):
        file = download_file(MODEL_URL, path)
        tar = tarfile.open(file)
        info_list = tar.getmembers()
        idx = [os.path.splitext(info.name)[1].lower() for info in info_list].index('.onnx')
        onnx_info = info_list[idx]
        tar.extract(onnx_info, path)
        onnx_file = os.path.join(path, onnx_info.name.replace('/', os.path.sep))
        shutil.copyfile(onnx_file, os.path.join(path, MODEL_FILE))
        tar.close()
    else:
        print('{} already exists'.format(model_file))

def create_config_file(path):
    config_file = os.path.join(path, CONFIG_FILE)
    with open(config_file, 'w') as f:
        f.write(CONFIG_DATA)

def download_label(path):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Downloading the label file from {}'.format(LABEL_URL))
    download_file(LABEL_URL, path)

def download_model(path):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Downloading the ONNX model file from {}'.format(MODEL_URL))
    extract_onnx_file(path)
    print('Creating the config file {}'.format(CONFIG_FILE))
    create_config_file(path)
    print('Model downloaded successfully')

def main():
    model_path = os.path.join(os.getcwd(), 'model')
    download_model(model_path)

if __name__ == '__main__':
    main()