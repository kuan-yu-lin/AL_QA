#!/bin/bash

python main.py \
    --exp_id 524a \
    -a RandomSampling \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -u True \
    -r True \
    -g 7

python main.py \
    --exp_id 524b \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -u True \
    -r True \
    -g 7

python main.py \
    --exp_id 524e \
    -a MarginSamplingDropout \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -u True \
    -r True \
    -g 7

python main.py \
    --exp_id 524j \
    -a MeanSTD \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -u True \
    -r True \
    -g 7

python main.py \
    --exp_id 524h \
    -a KMeansSampling \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -u True \
    -r True \
    -g 7

python main.py \
    --exp_id 524c \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -u True \
    -r True \
    -g 7

python main.py \
    --exp_id 524f \
    -a LeastConfidenceDropout \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -u True \
    -r True \
    -g 7

python main.py \
    --exp_id 524k \
    -a BALDDropout \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -u True \
    -r True \
    -g 7

python main.py \
    --exp_id 524i \
    -a KCenterGreedy \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -u True \
    -r True \
    -g 7

python main.py \
    --exp_id 524d \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -u True \
    -r True \
    -g 7

python main.py \
    --exp_id 524g \
    -a EntropySamplingDropout \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -u True \
    -r True \
    -g 7

python main.py \
    --exp_id 524l \
    -a BadgeSampling \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -u True \
    -r True \
    -g 8

