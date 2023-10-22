#!/bin/bash

python main.py \
    -a RandomSampling \
    -m Bert \
    -g 8

python main.py \
    -a MarginSampling \
    -m Bert \
    -g 8

python main.py \
    -a LeastConfidence \
    -m Bert \
    -g 8

python main.py \
    -a EntropySampling \
    -m Bert \
    -g 8

python main.py \
    -a MarginSamplingDropout \
    -m Bert \
    -g 8

python main.py \
    -a LeastConfidenceDropout \
    -m Bert \
    -g 8

python main.py \
    -a EntropySamplingDropout \
    -m Bert \
    -g 8

python main.py \
    -a KMeansSampling \
    -m Bert \
    -g 8

python main.py \
    -a KCenterGreedy \
    -m Bert \
    -g 8

python main.py \
    -a BALDDropout \
    -m Bert \
    -g 8

python main.py \
    -a MeanSTD \
    -m Bert \
    -g 8

python main.py \
    -a BadgeSampling \
    -m Bert \
    -g 8

python main.py \
    -a BatchBALDDropout \
    -m Bert \
    -g 8
