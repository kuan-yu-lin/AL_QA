#!/bin/bash

python main.py \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d DROP \
    -m RoBERTa \
    -r True \
    -g 4

python main.py \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m RoBERTa \
    -r True \
    -g 5

python main.py \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -m RoBERTa \
    -r True \
    -g 4

python main.py \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m RoBERTa \
    -r True \
    -g 6

python main.py \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -r True \
    -g 3

python main.py \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d DROP \
    -m RoBERTa \
    -r True \
    -g 6

python main.py \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m RoBERTa \
    -r True \
    -g 6

python main.py \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -m RoBERTa \
    -r True \
    -g 6

python main.py \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m RoBERTa \
    -r True \
    -g 7

python main.py \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -r True \
    -g 5

python main.py \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d DROP \
    -m RoBERTa \
    -r True \
    -g 3

python main.py \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m RoBERTa \
    -r True \
    -g 2

python main.py \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -m RoBERTa \
    -r True \
    -g 3

python main.py \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m RoBERTa \
    -r True \
    -g 8

python main.py \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -r True \
    -g 2

python main.py \
    -a MarginSamplingDropout \
    -q 200 \
    -b 50 \
    -d DROP \
    -m RoBERTa \
    -r True \
    -g 3

python main.py \
    -a MarginSamplingDropout \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m RoBERTa \
    -r True \
    -g 5

python main.py \
    -a MarginSamplingDropout \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -m RoBERTa \
    -r True \
    -g 6

python main.py \
    -a MarginSamplingDropout \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m RoBERTa \
    -r True \
    -g 3

python main.py \
    -a MarginSamplingDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -r True \
    -g 5

python main.py \
    -a LeastConfidenceDropout \
    -q 200 \
    -b 50 \
    -d DROP \
    -m RoBERTa \
    -r True \
    -g 8

python main.py \
    -a LeastConfidenceDropout \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m RoBERTa \
    -r True \
    -g 3

python main.py \
    -a LeastConfidenceDropout \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -m RoBERTa \
    -r True \
    -g 7

python main.py \
    -a LeastConfidenceDropout \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m RoBERTa \
    -r True \
    -g 6

python main.py \
    -a LeastConfidenceDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -r True \
    -g 7

python main.py \
    -a EntropySamplingDropout \
    -q 200 \
    -b 50 \
    -d DROP \
    -m RoBERTa \
    -r True \
    -g 3

python main.py \
    -a EntropySamplingDropout \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m RoBERTa \
    -r True \
    -g 8

python main.py \
    -a EntropySamplingDropout \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -m RoBERTa \
    -r True \
    -g 5

python main.py \
    -a EntropySamplingDropout \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m RoBERTa \
    -r True \
    -g 8

python main.py \
    -a EntropySamplingDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -r True \
    -g 2
