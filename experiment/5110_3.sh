#!/bin/bash

python main.py \
    --exp_id 511a \
    -a RandomSampling \
    -u True \
    -g 5

python main.py \
    --exp_id 511d \
    -a EntropySampling \
    -u True \
    -g 4

python main.py \
    --exp_id 511g \
    -a EntropySamplingDropout \
    -u True \
    -g 4

python main.py \
    --exp_id 511l \
    -a BadgeSampling \
    -u True \
    -g 4