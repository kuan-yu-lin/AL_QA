#!/bin/bash

# python main.py \
#     --exp_id 1 \
#     -a RandomSampling \
#     -q 200 \
#     -b 50 \
#     -d BioASQ \
#     -u True \
#     -r True \
#     -g 2

# python main.py \
#     --exp_id 2 \
#     -a MarginSampling \
#     -q 200 \
#     -b 50 \
#     -d BioASQ \
#     -u True \
#     -r True \
#     -g 2

python main.py \
    --exp_id 3 \
    -a MarginSamplingDropout \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 2

python main.py \
    --exp_id 4 \
    -a MeanSTD \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 2
