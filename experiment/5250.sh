#!/bin/bash

python main.py \
    --exp_id 525a \
    -a RandomSampling \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -u True \
    -r True \
    -g 6

python main.py \
    --exp_id 525b \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -u True \
    -r True \
    -g 6

python main.py \
    --exp_id 525e \
    -a MarginSamplingDropout \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -u True \
    -r True \
    -g 6

python main.py \
    --exp_id 525j \
    -a MeanSTD \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -u True \
    -r True \
    -g 6

python main.py \
    --exp_id 525h \
    -a KMeansSampling \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -u True \
    -r True \
    -g 6

python main.py \
    --exp_id 525c \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -u True \
    -r True \
    -g 6

python main.py \
    --exp_id 525f \
    -a LeastConfidenceDropout \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -u True \
    -r True \
    -g 6

python main.py \
    --exp_id 525k \
    -a BALDDropout \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -u True \
    -r True \
    -g 6

python main.py \
    --exp_id 525i \
    -a KCenterGreedy \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -u True \
    -r True \
    -g 6

python main.py \
    --exp_id 525d \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -u True \
    -r True \
    -g 6

python main.py \
    --exp_id 525g \
    -a EntropySamplingDropout \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -u True \
    -r True \
    -g 6

python main.py \
    --exp_id 525l \
    -a BadgeSampling \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -u True \
    -r True \
    -g 6