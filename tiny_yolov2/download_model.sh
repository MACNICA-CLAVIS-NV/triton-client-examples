#!/usr/bin/env bash

set -eu

MODEL_URL="https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.tar.gz"
LABEL_URL="https://raw.githubusercontent.com/pjreddie/darknet/master/data/voc.names"

MODEL_FILE=`basename ${MODEL_URL}`
LABEL_FILE=`basename ${LABEL_URL}`

REPOSITORY_PATH=${PWD}/model_repository
if [ $# -gt 0 ]; then
    REPOSITORY_PATH=$1
fi

MODEL_NAME="tinyyolov2_onnx"
MODEL_VERSION="1"
MODEL_PATH=${REPOSITORY_PATH}/${MODEL_NAME}
VERSION_PATH=${MODEL_PATH}/${MODEL_VERSION}
MODEL_CACHE=".model_cache"

echo "Repository Path: ${REPOSITORY_PATH}"

mkdir -p ${VERSION_PATH}

# Config file
cp config.pbtxt ${MODEL_PATH}

# Label file
wget ${LABEL_URL} -O ${MODEL_PATH}/${LABEL_FILE}

# Model file
mkdir -p ${MODEL_CACHE}
wget ${MODEL_URL} -O ${MODEL_CACHE}/${MODEL_FILE}
tar -xvozf ${MODEL_CACHE}/${MODEL_FILE} -C ${MODEL_CACHE}
cp ${MODEL_CACHE}/tiny_yolov2/Model.onnx ${VERSION_PATH}/model.onnx
rm -rf ${MODEL_CACHE}
