#!/bin/bash

python main.py \
    -a RandomSampling \
    -g 3

python main.py \
    -a MarginSampling \
    -g 3

python main.py \
    -a LeastConfidence \
    -g 8

python main.py \
    -a EntropySampling \
    -g 3

python main.py \
    -a MarginSamplingDropout \
    -g 4

python main.py \
    -a LeastConfidenceDropout \
    -g 5

python main.py \
    -a EntropySamplingDropout \
    -g 6

python main.py \
    -a KMeansSampling \
    -g 3

python main.py \
    -a KCenterGreedy \
    -g 4

python main.py \
    -a BALDDropout \
    -g 3

python main.py \
    -a MeanSTD \
    -g 3

python main.py \
    -a BadgeSampling \
    -g 2
