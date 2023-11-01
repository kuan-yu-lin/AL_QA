#!/bin/bash

python main.py \
    --exp_id  \
    -a KCenterGreedy \
    -u True \
    -g 4

python main.py \
    --exp_id  \
    -a BALDDropout \
    -u True \
    -g 4

python main.py \
    --exp_id  \
    -a LeastConfidence \
    -u True \
    -g 4

python main.py \
    --exp_id  \
    -a LeastConfidenceDropout \
    -u True \
    -g 4
