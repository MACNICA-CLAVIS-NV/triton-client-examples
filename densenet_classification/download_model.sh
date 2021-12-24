#!/usr/bin/env bash

set -eu

MODEL_URL="https://contentmamluswest001.blob.core.windows.net/content/14b2744cf8d6418c87ffddc3f3127242/9502630827244d60a1214f250e3bbca7/08aed7327d694b8dbaee2c97b8d0fcba/densenet121-1.2.onnx"
LABEL_URL="https://raw.githubusercontent.com/triton-inference-server/server/main/docs/examples/model_repository/densenet_onnx/densenet_labels.txt"
CONFIG_URL="https://raw.githubusercontent.com/triton-inference-server/server/main/docs/examples/model_repository/densenet_onnx/config.pbtxt"

MODEL_FILE="model.onnx"
LABEL_FILE=`basename ${LABEL_URL}`
CONFIG_FILE=`basename ${CONFIG_URL}`

REPOSITORY_PATH=${PWD}/model_repository
if [ $# -gt 0 ]; then
    REPOSITORY_PATH=$1
fi

MODEL_NAME="densenet_onnx"
MODEL_VERSION="1"
MODEL_PATH=${REPOSITORY_PATH}/${MODEL_NAME}
VERSION_PATH=${MODEL_PATH}/${MODEL_VERSION}

echo "Repository Path: ${REPOSITORY_PATH}"

mkdir -p ${VERSION_PATH}

# Config file
wget ${CONFIG_URL} -O ${MODEL_PATH}/${CONFIG_FILE}

# Label file
wget ${LABEL_URL} -O ${MODEL_PATH}/${LABEL_FILE}

# Model file
wget ${MODEL_URL} -O ${VERSION_PATH}/${MODEL_FILE}
