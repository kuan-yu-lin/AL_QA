# ####################
# Model: Bert-base
#
# source domain: SQuAD, Natural Question
#
# target domain: BioASQ, NewsQA, SearchQA, TextbookQA, DROP
#
# Init pool: 50
# Query: 50, 100, 150, 200
# Query batch: 50 
# Experiment Iteration: 5
# ####################

# Pretrain
python pretrain.py \
    -d SQuAD \
    -m Bert \
    -e 1127 \
    -l 3e-5 \
    # -g 3

# Random
## 7/13 exp1 waldweihe 2
python main_lowRes.py \
    -a RandomSampling \
    -s 50 \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m Bert \
    -e 1127 \
    -l 3e-5 \
    -t 5 \
    -g 2

# Margin
## 7/13 exp1 waldweihe 3
python main_lowRes.py \
    -a MarginSampling \
    -s 50 \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m Bert \
    -e 1127 \
    -l 3e-5 \
    -t 5 \
    -g 3

# LC
## 7/13 exp1 strauss 1
## iter. = 3 # wrong =.=
python main_lowRes.py \
    -a LeastConfidence \
    -s 50 \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m Bert \
    -e 1127 \
    -l 3e-5 \
    -t 5 \
    -g 1

# Entropy
## 7/13 exp1 strauss 8
## iter. = 3 # wrong =.=
python main_lowRes.py \
    -a EntropySampling \
    -s 50 \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m Bert \
    -e 1127 \
    -l 3e-5 \
    -t 5 \
    -g 8

# MarginDropout
python main_lowRes.py \
    -a MarginSamplingDropout \
    -s 50 \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m Bert \
    -e 1127 \
    -l 3e-5 \
    -t 5 \
    -g 3

# LCDropout
python main_lowRes.py \
    -a LeastConfidenceDropout \
    -s 50 \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m Bert \
    -e 1127 \
    -l 3e-5 \
    -t 5 \
    -g 2

# EntropyDropout
python main_lowRes.py \
    -a EntropySamplingDropout \
    -s 50 \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m Bert \
    -e 1127 \
    -l 3e-5 \
    -t 5 \
    -g 2

# KMeans
python main_lowRes.py \
    -a KMeansSampling \
    -s 50 \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m Bert \
    -e 1127 \
    -l 3e-5 \
    -t 5 \
    -g 3

# KCenterGreedy
python main_lowRes.py \
    -a KCenterGreedy \
    -s 50 \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m Bert \
    -e 1127 \
    -l 3e-5 \
    -t 5 \
    -g 2

# Bayesian
python main_lowRes.py \
    -a BALDDropout \
    -s 50 \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m Bert \
    -e 1127 \
    -l 3e-5 \
    -t 5 \
    -g 3

# MeanSTD
python main_lowRes.py \
    -a MeanSTD \
    -s 50 \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m Bert \
    -e 1127 \
    -l 3e-5 \
    -t 5 \
    -g 2

# BADGE
python main_lowRes.py \
    -a BadgeSampling \
    -s 50 \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m Bert \
    -e 1127 \
    -l 3e-5 \
    -t 5 \
    -g 0

# LPL
python main_lowRes.py \
    -a LossPredictionLoss \
    -s 50 \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m Bert \
    -e 1127 \
    -l 3e-5 \
    -t 5 \
    -g 2

# CEAL
python main_lowRes.py \
    -a CEALSampling \
    -s 50 \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m Bert \
    -e 1127 \
    -l 3e-5 \
    -t 5 \
    -g 2