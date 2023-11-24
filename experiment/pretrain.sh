#!/bin/bash

python pretrain.py \
    -m RoBERTaLarge \
    -g 4

python pretrain.py \
    -m BERTLarge \
    -g 4

python pretrain.py \
    -m RoBERTa \
    -g 4

python pretrain.py \
    -m Bert \
    -g 4
