#!/bin/bash

python main.py \
    -a RandomSampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTaLarge \
    -r True \
    -g 4

python main.py \
    -a KMeansSampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTaLarge \
    -r True \
    -g 4

python main.py \
    -a KCenterGreedy \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTaLarge \
    -r True \
    -g 4

python main.py \
    -a BALDDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTaLarge \
    -r True \
    -g 4

python main.py \
    -a MeanSTD \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTaLarge \
    -r True \
    -g 4

python main.py \
    -a BadgeSampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTaLarge \
    -r True \
    -g 4