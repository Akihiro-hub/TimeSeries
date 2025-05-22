#!/bin/bash
# setup.sh ファイル

# 必要なシステムパッケージをインストール
apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    python3-dev \
    gfortran \
    libatlas-base-dev \
    libfreetype6-dev \
    libpng-dev \
    libhdf5-serial-dev \
    libhdf5-dev \
    python3-tk \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージのインストール
pip install --upgrade pip
pip install -r requirements.txt
