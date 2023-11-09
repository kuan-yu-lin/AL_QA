#!/bin/bash

python main.py \
    --exp_id 123a \
    --exp_round 1 \
    -a RandomSampling \
    -q 50 \
    -b 50 \
    -d DROP \
    -r True \
    -g 5

python main.py \
    --exp_id 124a \
    --exp_round 1 \
    -a RandomSampling \
    -q 50 \
    -b 50 \
    -d TextbookQA \
    -r True \
    -g 5

# python main.py \
#     --exp_id 122a \
#     -a RandomSampling \
#     -q 200 \
#     -b 50 \
#     -d BioASQ \
#     -r True \
#     -g 5



# python main.py \
#     --exp_id 12 \
#     -a RandomSampling \
#     -q 200 \
#     -b 50 \
#     -d NewsQA \
#     -r True \
#     -g 3

# python main.py \
#     --exp_id 12 \
#     -a RandomSampling \
#     -q 200 \
#     -b 50 \
#     -d SearchQA \
#     -r True \
#     -g 6

# python main.py \
#     --exp_id 12 \
#     -a KMeansSampling \
#     -q 200 \
#     -b 50 \
#     -d DROP \
#     -r True \
#     -g 6

# python main.py \
#     --exp_id 12 \
#     -a KMeansSampling \
#     -q 200 \
#     -b 50 \
#     -d BioASQ \
#     -r True \
#     -g 6

# python main.py \
#     --exp_id 12 \
#     -a KMeansSampling \
#     -q 200 \
#     -b 50 \
#     -d TextbookQA \
#     -r True \
#     -g 7

# python main.py \
#     --exp_id 12 \
#     -a KMeansSampling \
#     -q 200 \
#     -b 50 \
#     -d NewsQA \
#     -r True \
#     -g 3

# python main.py \
#     --exp_id 12 \
#     -a KMeansSampling \
#     -q 200 \
#     -b 50 \
#     -d SearchQA \
#     -r True \
#     -g 5

# python main.py \
#     --exp_id 12 \
#     -a KCenterGreedy \
#     -q 200 \
#     -b 50 \
#     -d DROP \
#     -r True \
#     -g 3

# python main.py \
#     --exp_id 12 \
#     -a KCenterGreedy \
#     -q 200 \
#     -b 50 \
#     -d BioASQ \
#     -r True \
#     -g 8

# python main.py \
#     --exp_id 12 \
#     -a KCenterGreedy \
#     -q 200 \
#     -b 50 \
#     -d TextbookQA \
#     -r True \
#     -g 3

# python main.py \
#     --exp_id 12 \
#     -a KCenterGreedy \
#     -q 200 \
#     -b 50 \
#     -d NewsQA \
#     -r True \
#     -g 6

# python main.py \
#     --exp_id 12 \
#     -a KCenterGreedy \
#     -q 200 \
#     -b 50 \
#     -d SearchQA \
#     -r True \
#     -g 7

# python main.py \
#     --exp_id 12 \
#     -a BALDDropout \
#     -q 200 \
#     -b 50 \
#     -d DROP \
#     -r True \
#     -g 3

# python main.py \
#     --exp_id 12 \
#     -a BALDDropout \
#     -q 200 \
#     -b 50 \
#     -d BioASQ \
#     -r True \
#     -g 5

# python main.py \
#     --exp_id 12 \
#     -a BALDDropout \
#     -q 200 \
#     -b 50 \
#     -d TextbookQA \
#     -r True \
#     -g 8

# python main.py \
#     --exp_id 12 \
#     -a BALDDropout \
#     -q 200 \
#     -b 50 \
#     -d NewsQA \
#     -r True \
#     -g 8

# python main.py \
#     --exp_id 12 \
#     -a BALDDropout \
#     -q 200 \
#     -b 50 \
#     -d SearchQA \
#     -r True \
#     -g 2

# python main.py \
#     --exp_id 12 \
#     -a MeanSTD \
#     -q 200 \
#     -b 50 \
#     -d DROP \
#     -r True \
#     -g 8

# python main.py \
#     --exp_id 12 \
#     -a MeanSTD \
#     -q 200 \
#     -b 50 \
#     -d BioASQ \
#     -r True \
#     -g 7

# python main.py \
#     --exp_id 12 \
#     -a MeanSTD \
#     -q 200 \
#     -b 50 \
#     -d TextbookQA \
#     -r True \
#     -g 3

# python main.py \
#     --exp_id 12 \
#     -a MeanSTD \
#     -q 200 \
#     -b 50 \
#     -d NewsQA \
#     -r True \
#     -g 7

# python main.py \
#     --exp_id 12 \
#     -a MeanSTD \
#     -q 200 \
#     -b 50 \
#     -d SearchQA \
#     -r True \
#     -g 5

# python main.py \
#     --exp_id 12 \
#     -a BadgeSampling \
#     -q 200 \
#     -b 50 \
#     -d DROP \
#     -r True \
#     -g 8

# python main.py \
#     --exp_id 12 \
#     -a BadgeSampling \
#     -q 200 \
#     -b 50 \
#     -d BioASQ \
#     -r True \
#     -g 7

# python main.py \
#     --exp_id 12 \
#     -a BadgeSampling \
#     -q 200 \
#     -b 50 \
#     -d TextbookQA \
#     -r True \
#     -g 5

# python main.py \
#     --exp_id 12 \
#     -a BadgeSampling \
#     -q 200 \
#     -b 50 \
#     -d NewsQA \
#     -r True \
#     -g 3

# python main.py \
#     --exp_id 12 \
#     -a BadgeSampling \
#     -q 200 \
#     -b 50 \
#     -d SearchQA \
#     -r True \
#     -g 6
