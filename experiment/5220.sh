#!/bin/bash

python main.py \
    --exp_id 522a \
    -a RandomSampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 6

python main.py \
    --exp_id 522b \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 6

python main.py \
    --exp_id 522e \
    -a MarginSamplingDropout \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 6

python main.py \
    --exp_id 522j \
    -a MeanSTD \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 6

python main.py \
    --exp_id 522h \
    -a KMeansSampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 6

python main.py \
    --exp_id 522c \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 6

python main.py \
    --exp_id 522f \
    -a LeastConfidenceDropout \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 6

python main.py \
    --exp_id 522k \
    -a BALDDropout \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 6

python main.py \
    --exp_id 522i \
    -a KCenterGreedy \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 6

python main.py \
    --exp_id 522d \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 6

python main.py \
    --exp_id 522g \
    -a EntropySamplingDropout \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 6

python main.py \
    --exp_id 522l \
    -a BadgeSampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 6