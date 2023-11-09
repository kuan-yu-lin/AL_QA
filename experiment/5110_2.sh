#!/bin/bash

python main.py \
    --exp_id 511i \
    -a KCenterGreedy \
    -u True \
    -g 7

python main.py \
    --exp_id 511k \
    -a BALDDropout \
    -u True \
    -g 7

python main.py \
    --exp_id 511c \
    -a LeastConfidence \
    -u True \
    -g 7

python main.py \
    --exp_id 511f \
    -a LeastConfidenceDropout \
    -u True \
    -g 7
