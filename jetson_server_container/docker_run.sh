#!/usr/bin/env bash

set -eu

source docker_base.sh

MODEL_REPO_HOST_PATH=${PWD}/model_repository
MODEL_REPO_CONTAINER_PATH=/triton/model_repository

mkdir -p ${MODEL_REPO_HOST_PATH}

sudo docker run \
    -it \
    --rm \
    --net=host \
    --runtime nvidia \
    --gpus=1 \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -v ${MODEL_REPO_HOST_PATH}:${MODEL_REPO_CONTAINER_PATH} \
    jetson-triton:l4t-r${L4T_VERSION} \
    ./bin/tritonserver --model-repository=${MODEL_REPO_CONTAINER_PATH} --backend-directory=./backends --backend-config=tensorflow,version=2 --model-control-mode=explicit
