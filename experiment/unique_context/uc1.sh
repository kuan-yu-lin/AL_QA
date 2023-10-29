#!/bin/bash

python main.py \
    --exp_id  \
    -a KMeansSampling \
    -u True \
    -g 3

python main.py \
    --exp_id  \
    -a MarginSampling \
    -u True \
    -g 3

python main.py \
    --exp_id  \
    -a MarginSamplingDropout \
    -u True \
    -g 3

python main.py \
    --exp_id  \
    -a MeanSTD \
    -u True \
    -g 3