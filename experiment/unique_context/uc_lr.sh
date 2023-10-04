#!/bin/bash

python main.py \
    -a RandomSampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 7

python main.py \
    -a KMeansSampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 7

python main.py \
    -a KCenterGreedy \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 7

python main.py \
    -a BALDDropout \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 7

python main.py \
    -a MeanSTD \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 7

python main.py \
    -a BadgeSampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 7

python main.py \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 7

python main.py \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 7

python main.py \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 7

python main.py \
    -a MarginSamplingDropout \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 7

python main.py \
    -a LeastConfidenceDropout \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 7

python main.py \
    -a EntropySamplingDropout \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 7
