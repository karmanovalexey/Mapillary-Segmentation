#FROM ubuntu:18.04
FROM nvidia/cudagl:10.0-devel-ubuntu18.04
MAINTAINER naverlabs "kapture@naverlabs.com"

# setup environment
ENV     LANG C.UTF-8
ENV     LC_ALL C.UTF-8
ENV     DEBIAN_FRONTEND noninteractive
# arguments
ARG     MAKE_OPTIONS="-j8"
ARG     SOURCE_PREFIX="/opt/src"

RUN mkdir -p ${SOURCE_PREFIX}

# Get dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    git \
    wget\
    curl \
    python3.6 python3-pip \
    pandoc asciidoctor \
    cmake \
    build-essential \
    libboost-all-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    freeglut3-dev \
    libxmu-dev \
    libxi-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    libcgal-qt5-dev \
    libqt5opengl5-dev \
    qt5-default \
    x11-apps \
    mesa-utils \
  && rm -rf /var/lib/apt/lists/*

RUN 	 apt-get update  \
	 && apt-get install -y \
         zip unzip


### FINALIZE ###################################################################
# save space: purge apt-get
RUN     rm -rf /var/lib/apt/lists/*
