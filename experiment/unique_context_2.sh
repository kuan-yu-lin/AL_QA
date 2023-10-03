#!/bin/bash

python main.py \
    -a KCenterGreedy \
    -u True \
    -g 4

python main.py \
    -a BALDDropout \
    -u True \
    -g 4

python main.py \
    -a LeastConfidence \
    -u True \
    -g 4

python main.py \
    -a LeastConfidenceDropout \
    -u True \
    -g 4
