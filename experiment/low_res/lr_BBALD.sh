#!/bin/bash

python main.py \
    -a BatchBALDDropout \
    -q 200 \
    -b 50 \
    -d DROP \
    -r True \
    -g 5

python main.py \
    -a BatchBALDDropout \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -r True \
    -g 5

python main.py \
    -a BatchBALDDropout \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -r True \
    -g 5

python main.py \
    -a BatchBALDDropout \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -r True \
    -g 5

python main.py \
    -a BatchBALDDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -r True \
    -g 5
