#!/bin/bash

python main.py \
    --exp_id \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d DROP \
    -r True \
    -g 4

python main.py \
    --exp_id 122b \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -r True \
    -g 5

python main.py \
    --exp_id \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -r True \
    -g 4

python main.py \
    --exp_id \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -r True \
    -g 6

python main.py \
    --exp_id \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -r True \
    -g 3

python main.py \
    --exp_id \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d DROP \
    -r True \
    -g 6

python main.py \
    --exp_id \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -r True \
    -g 6

python main.py \
    --exp_id \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -r True \
    -g 6

python main.py \
    --exp_id \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -r True \
    -g 7

python main.py \
    --exp_id \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -r True \
    -g 5

python main.py \
    --exp_id \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d DROP \
    -r True \
    -g 3

python main.py \
    --exp_id \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -r True \
    -g 2

python main.py \
    --exp_id \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -r True \
    -g 3

python main.py \
    --exp_id \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -r True \
    -g 8

python main.py \
    --exp_id \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -r True \
    -g 2

python main.py \
    --exp_id \
    -a MarginSamplingDropout \
    -q 200 \
    -b 50 \
    -d DROP \
    -r True \
    -g 3

python main.py \
    --exp_id \
    -a MarginSamplingDropout \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -r True \
    -g 5

python main.py \
    --exp_id \
    -a MarginSamplingDropout \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -r True \
    -g 6

python main.py \
    --exp_id \
    -a MarginSamplingDropout \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -r True \
    -g 3

python main.py \
    --exp_id \
    -a MarginSamplingDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -r True \
    -g 5

python main.py \
    --exp_id \
    -a LeastConfidenceDropout \
    -q 200 \
    -b 50 \
    -d DROP \
    -r True \
    -g 8

python main.py \
    --exp_id \
    -a LeastConfidenceDropout \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -r True \
    -g 3

python main.py \
    --exp_id \
    -a LeastConfidenceDropout \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -r True \
    -g 7

python main.py \
    --exp_id \
    -a LeastConfidenceDropout \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -r True \
    -g 6

python main.py \
    --exp_id \
    -a LeastConfidenceDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -r True \
    -g 7

python main.py \
    --exp_id \
    -a EntropySamplingDropout \
    -q 200 \
    -b 50 \
    -d DROP \
    -r True \
    -g 3

python main.py \
    --exp_id \
    -a EntropySamplingDropout \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -r True \
    -g 8

python main.py \
    --exp_id \
    -a EntropySamplingDropout \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -r True \
    -g 5

python main.py \
    --exp_id \
    -a EntropySamplingDropout \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -r True \
    -g 8

python main.py \
    --exp_id \
    -a EntropySamplingDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -r True \
    -g 2
