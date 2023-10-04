#!/bin/bash

python main.py \
    -a RandomSampling \
    -u True \
    -g 5

python main.py \
    -a EntropySampling \
    -u True \
    -g 5

python main.py \
    -a EntropySamplingDropout \
    -u True \
    -g 5

python main.py \
    -a BadgeSampling \
    -u True \
    -g 5