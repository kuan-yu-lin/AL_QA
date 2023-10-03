#!/bin/bash

python main.py \
    -a KMeansSampling \
    -u True \
    -g 3

python main.py \
    -a MeanSTD \
    -u True \
    -g 3

python main.py \
    -a MarginSampling \
    -u True \
    -g 3

python main.py \
    -a MarginSamplingDropout \
    -u True \
    -g 3