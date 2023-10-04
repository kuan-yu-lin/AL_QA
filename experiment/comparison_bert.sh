#####################
# Model: Bert-base (fixed)
#
# Data: SQuAD
#
# Init pool: 500
# Query: 500, 1000, 1500, 2000 (mention it the proposal)
# Query batch: 500
# Experiment Iteration: 5
#####################

# Random
## 9/22 strauss
python main.py \
    -a RandomSampling \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m Bert \
    -e 1127 \
    -p 5 \
    -g 8

# Margin
## 9/22 strauss
python main.py \
    -a MarginSampling \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m Bert \
    -e 1127 \
    -p 5 \
    -g 4

# LC
## 9/22 strauss
python main.py \
    -a LeastConfidence \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m Bert \
    -e 1127 \
    -p 5 \
    -g 5

# Entropy
## 9/22 strauss
python main.py \
    -a EntropySampling \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m Bert \
    -e 1127 \
    -p 5 \
    -g 6

# MarginDropout
## 9/22 strauss (fail)
## 9/27 - 10/3 strauss
python main.py \
    -a MarginSamplingDropout \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m Bert \
    -e 1127 \
    -p 5 \
    -g 3

# LCDropout
## 9/22 strauss (fail)
## 9/27 - 10/3 strauss
python main.py \
    -a LeastConfidenceDropout \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m Bert \
    -e 1127 \
    -p 5 \
    -g 4

# EntropyDropout
## 9/22 strauss (fail)
## 9/27 - 10/3 strauss
python main.py \
    -a EntropySamplingDropout \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m Bert \
    -e 1127 \
    -p 5 \
    -g 5

# KMeans
## 9/22 strauss
python main.py \
    -a KMeansSampling \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m Bert \
    -e 1127 \
    -p 5 \
    -g 7

# KCenterGreedy
## 9/22 strauss
python main.py \
    -a KCenterGreedy \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m Bert \
    -e 1127 \
    -p 5 \
    -g 3

# Bayesian/BALD
## 10/3 strauss
python main.py \
    -a BALDDropout \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m Bert \
    -e 1127 \
    -p 5 \
    -g 8

# MeanSTD
## 9/22 strauss (fail)
## 9/27 - 10/3 strauss
python main.py \
    -a MeanSTD \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m Bert \
    -e 1127 \
    -p 5 \
    -g 8

# BADGE
## 10/3 kapweihe
python main.py \
    -a BadgeSampling \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m Bert \
    -e 1127 \
    -p 5 \
    -g 3
