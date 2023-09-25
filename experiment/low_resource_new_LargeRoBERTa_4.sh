#!/bin/bash

python main_lowRes_new.py \
    -a RandomSampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTaLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 4

python main_lowRes_new.py \
    -a KMeansSampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTaLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 4

python main_lowRes_new.py \
    -a KCenterGreedy \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTaLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 4

python main_lowRes_new.py \
    -a BALDDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTaLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 4

python main_lowRes_new.py \
    -a MeanSTD \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTaLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 4

python main_lowRes_new.py \
    -a BadgeSampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTaLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 4
