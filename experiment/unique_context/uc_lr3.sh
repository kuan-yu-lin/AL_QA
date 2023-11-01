#!/bin/bash

python main.py \
    --exp_id 122i \
    -a KCenterGreedy \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 4

python main.py \
    --exp_id 10 \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 4

python main.py \
    --exp_id 11 \
    -a EntropySamplingDropout \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 4

python main.py \
    --exp_id 12 \
    -a BadgeSampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 4
