#!/bin/bash

python main.py \
    --exp_id 13 \
    -a RandomSampling \
    -u True \
    -g 5

python main.py \
    --exp_id 14 \
    -a EntropySampling \
    -u True \
    -g 5

python main.py \
    --exp_id 15 \
    -a EntropySamplingDropout \
    -u True \
    -g 5

python main.py \
    --exp_id 16 \
    -a BadgeSampling \
    -u True \
    -g 5