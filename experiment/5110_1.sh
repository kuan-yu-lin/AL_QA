#!/bin/bash

python main.py \
    --exp_id 511h \
    -a KMeansSampling \
    -u True \
    -g 8

python main.py \
    --exp_id 511b \
    -a MarginSampling \
    -u True \
    -g 8

python main.py \
    --exp_id 511e \
    -a MarginSamplingDropout \
    -u True \
    -g 8

python main.py \
    --exp_id 511j \
    -a MeanSTD \
    -u True \
    -g 8