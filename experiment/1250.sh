#!/bin/bash
# update: the train and val size of NewsQA

# python main.py \
#     --exp_id 125a \
#     -a RandomSampling \
#     -q 200 \
#     -b 50 \
#     -d NewsQA \
#     -r True \
#     -g 2

# python main.py \
#     --exp_id 125b \
#     -a MarginSampling \
#     -q 200 \
#     -b 50 \
#     -d NewsQA \
#     -r True \
#     -g 2

# python main.py \
#     --exp_id 125c \
#     -a LeastConfidence \
#     -q 200 \
#     -b 50 \
#     -d NewsQA \
#     -r True \
#     -g 2

# python main.py \
#     --exp_id 125d \
#     -a EntropySampling \
#     -q 200 \
#     -b 50 \
#     -d NewsQA \
#     -r True \
#     -g 2

# python main.py \
#     --exp_id 125e \
#     -a MarginSamplingDropout \
#     -q 200 \
#     -b 50 \
#     -d NewsQA \
#     -r True \
#     -g 2

python main.py \
    --exp_id 125f \
    -a LeastConfidenceDropout \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -r True \
    -g 2

python main.py \
    --exp_id 125g \
    -a EntropySamplingDropout \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -r True \
    -g 2

python main.py \
    --exp_id 125h \
    -a KMeansSampling \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -r True \
    -g 2

python main.py \
    --exp_id 125i \
    -a KCenterGreedy \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -r True \
    -g 2

python main.py \
    --exp_id 125j \
    -a MeanSTD \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -r True \
    -g 2

python main.py \
    --exp_id 125k \
    -a BALDDropout \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -r True \
    -g 2

python main.py \
    --exp_id 125l \
    -a BadgeSampling \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -r True \
    -g 2