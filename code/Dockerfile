# Use Tensorflow from nvidia-docker image as parent image
FROM nvcr.io/nvidia/tensorflow:18.06-py3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
COPY . /tmp/
