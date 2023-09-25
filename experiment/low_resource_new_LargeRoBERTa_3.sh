#!/bin/bash

python main_lowRes_new.py \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTaLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

python main_lowRes_new.py \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTaLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

python main_lowRes_new.py \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTaLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

python main_lowRes_new.py \
    -a MarginSamplingDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTaLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

python main_lowRes_new.py \
    -a LeastConfidenceDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTaLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

python main_lowRes_new.py \
    -a EntropySamplingDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTaLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3