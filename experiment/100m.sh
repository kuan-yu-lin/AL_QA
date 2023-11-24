#!/bin/bash

# python main.py \
#     --exp_id 122m \
#     -a BatchBALDDropout \
#     -q 200 \
#     -b 50 \
#     -d BioASQ \
#     -r True \
#     -g 5

# python main.py \
#     --exp_id 123m \
#     -a BatchBALDDropout \
#     -q 200 \
#     -b 50 \
#     -d DROP \
#     -r True \
#     -g 5

python main.py \
    --exp_id 124m \
    -a BatchBALDDropout \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -r True \
    -g 5

# python main.py \
#     --exp_id 125m \
#     -a BatchBALDDropout \
#     -q 200 \
#     -b 50 \
#     -d NewsQA \
#     -r True \
#     -g 5

python main.py \
    --exp_id 126m \
    -a BatchBALDDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -r True \
    -g 5

# python main.py \
#     --exp_id 127m \
#     -a BatchBALDDropout \
#     -q 200 \
#     -b 50 \
#     -d NaturalQuestions \
#     -r True \
#     -g 5

# python main.py \
#     --exp_id 111m \
#     -a BadgeSampling \
#     -u True \
#     -g 5
