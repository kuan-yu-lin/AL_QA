#!/bin/bash

python main.py \
    --exp_id 5 \
    -a KMeansSampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 3

python main.py \
    --exp_id 6 \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 3

python main.py \
    --exp_id 7 \
    -a LeastConfidenceDropout \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 3

python main.py \
    --exp_id 8 \
    -a BALDDropout \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 3
