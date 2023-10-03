#!/bin/bash

python main_lowRes_new.py \
    -a RandomSampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -u True \
    -r True \
    -g 3

python main_lowRes_new.py \
    -a KMeansSampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -u True \
    -r True \
    -g 3

python main_lowRes_new.py \
    -a KCenterGreedy \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -u True \
    -r True \
    -g 3

python main_lowRes_new.py \
    -a BALDDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -u True \
    -r True \
    -g 3

python main_lowRes_new.py \
    -a MeanSTD \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -u True \
    -r True \
    -g 3

python main_lowRes_new.py \
    -a BadgeSampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -u True \
    -r True \
    -g 3

python main_lowRes_new.py \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -u True \
    -r True \
    -g 3

python main_lowRes_new.py \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -u True \
    -r True \
    -g 3

python main_lowRes_new.py \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -u True \
    -r True \
    -g 3

python main_lowRes_new.py \
    -a MarginSamplingDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -u True \
    -r True \
    -g 3

python main_lowRes_new.py \
    -a LeastConfidenceDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -u True \
    -r True \
    -g 3

python main_lowRes_new.py \
    -a EntropySamplingDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -u True \
    -r True \
    -g 3
