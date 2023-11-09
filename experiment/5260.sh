#!/bin/bash

python main.py \
    --exp_id 526a \
    -a RandomSampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -u True \
    -r True \
    -g 4

python main.py \
    --exp_id 526b \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -u True \
    -r True \
    -g 4

python main.py \
    --exp_id 526e \
    -a MarginSamplingDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -u True \
    -r True \
    -g 4

python main.py \
    --exp_id 526j \
    -a MeanSTD \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -u True \
    -r True \
    -g 4

python main.py \
    --exp_id 526h \
    -a KMeansSampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -u True \
    -r True \
    -g 4

python main.py \
    --exp_id 526c \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -u True \
    -r True \
    -g 4

python main.py \
    --exp_id 526f \
    -a LeastConfidenceDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -u True \
    -r True \
    -g 4

python main.py \
    --exp_id 526k \
    -a BALDDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -u True \
    -r True \
    -g 4

python main.py \
    --exp_id 526i \
    -a KCenterGreedy \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -u True \
    -r True \
    -g 4

python main.py \
    --exp_id 526d \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -u True \
    -r True \
    -g 4

python main.py \
    --exp_id 526g \
    -a EntropySamplingDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -u True \
    -r True \
    -g 4

python main.py \
    --exp_id 526l \
    -a BadgeSampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -u True \
    -r True \
    -g 4