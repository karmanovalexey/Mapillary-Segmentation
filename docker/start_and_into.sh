#!/bin/bash

orange=`tput setaf 3`
reset_color=`tput sgr0`

export ARCH=`uname -m`

echo "Running on ${orange}${ARCH}${reset_color}"

if [ "$ARCH" == "x86_64" ] 
then
    ARGS="--ipc host --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all"
elif [ "$ARCH" == "aarch64" ] 
then
    ARGS="--runtime nvidia"
else
    echo "Arch ${ARCH} not supported"
    exit
fi

dir_of_repo=/home/musaev_rv/repo
dir_of_dataset=$1

docker run -it --rm \
	--ipc host \
	--runtime nvidia \
	-e NVIDIA_DRIVER_CAPABILITIES=all \
        --env="DISPLAY=$DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --privileged \
        --name kapture \
        --net "host" \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	-v $dir_of_repo:/app:rw \
	-v /opt/src:/content:rw \
	-v $dir_of_dataset:/datasets:rw \
	--gpus all \
	-p 8889:8888 \
	hloc:latest

