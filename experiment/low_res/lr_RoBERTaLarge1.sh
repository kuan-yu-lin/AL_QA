#!/bin/bash

python main.py \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTaLarge \
    -r True \
    -g 3

python main.py \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTaLarge \
    -r True \
    -g 3

python main.py \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTaLarge \
    -r True \
    -g 3

python main.py \
    -a MarginSamplingDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTaLarge \
    -r True \
    -g 3

python main.py \
    -a LeastConfidenceDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTaLarge \
    -r True \
    -g 3

python main.py \
    -a EntropySamplingDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTaLarge \
    -r True \
    -g 3