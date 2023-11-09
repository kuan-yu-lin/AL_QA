#!/bin/bash

python main.py \
    --exp_id 523ht \
    -a KMeansSampling \
    -q 100 \
    -b 50 \
    -d DROP \
    -u True \
    -r True \
    -g 8

python main.py \
    --exp_id 523ct \
    -a LeastConfidence \
    -q 100 \
    -b 50 \
    -d DROP \
    -u True \
    -r True \
    -g 8

python main.py \
    --exp_id 524ht \
    -a KMeansSampling \
    -q 100 \
    -b 50 \
    -d TextbookQA \
    -u True \
    -r True \
    -g 8

python main.py \
    --exp_id 524ct \
    -a LeastConfidence \
    -q 100 \
    -b 50 \
    -d TextbookQA \
    -u True \
    -r True \
    -g 8
