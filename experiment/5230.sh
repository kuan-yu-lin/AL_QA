#!/bin/bash

python main.py \
    --exp_id 523a \
    -a RandomSampling \
    -q 200 \
    -b 50 \
    -d DROP \
    -u True \
    -r True \
    -g 5

python main.py \
    --exp_id 523b \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d DROP \
    -u True \
    -r True \
    -g 5

python main.py \
    --exp_id 523e \
    -a MarginSamplingDropout \
    -q 200 \
    -b 50 \
    -d DROP \
    -u True \
    -r True \
    -g 5

python main.py \
    --exp_id 523j \
    -a MeanSTD \
    -q 200 \
    -b 50 \
    -d DROP \
    -u True \
    -r True \
    -g 5

python main.py \
    --exp_id 523h \
    -a KMeansSampling \
    -q 200 \
    -b 50 \
    -d DROP \
    -u True \
    -r True \
    -g 5

python main.py \
    --exp_id 523c \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d DROP \
    -u True \
    -r True \
    -g 5

python main.py \
    --exp_id 523f \
    -a LeastConfidenceDropout \
    -q 200 \
    -b 50 \
    -d DROP \
    -u True \
    -r True \
    -g 5

python main.py \
    --exp_id 523k \
    -a BALDDropout \
    -q 200 \
    -b 50 \
    -d DROP \
    -u True \
    -r True \
    -g 5

python main.py \
    --exp_id 523i \
    -a KCenterGreedy \
    -q 200 \
    -b 50 \
    -d DROP \
    -u True \
    -r True \
    -g 5

python main.py \
    --exp_id 523d \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d DROP \
    -u True \
    -r True \
    -g 5

python main.py \
    --exp_id 523g \
    -a EntropySamplingDropout \
    -q 200 \
    -b 50 \
    -d DROP \
    -u True \
    -r True \
    -g 5

python main.py \
    --exp_id 523l \
    -a BadgeSampling \
    -q 200 \
    -b 50 \
    -d DROP \
    -u True \
    -r True \
    -g 5
