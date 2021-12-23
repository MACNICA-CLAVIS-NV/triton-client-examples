#!/usr/bin/env bash

set -eu

source docker_base.sh

cp /etc/apt/sources.list.d/nvidia-l4t-apt-source.list .
cp /etc/apt/trusted.gpg.d/jetson-ota-public.asc .

sudo docker build --build-arg BASE_IMAGE=${BASE_IMAGE} -t jetson-triton:l4t-r${L4T_VERSION} .
