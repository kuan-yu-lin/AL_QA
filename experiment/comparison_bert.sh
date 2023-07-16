#####################
# how well they work
# Model: Bert-base Bert-large(how long) RoBerta-large RoBerta-base(http://arxiv.org/abs/2101.00438)
#
# Init pool: 500
# Query: 500, 1000, 1500, 2000 (mention it the proposal)
# Query batch: 500
# Experiment Iteration: 5
#####################

# Random
## 7/13 exp1 waldweihe 2
## seed 4666
## 7/15 exp2 waldweihe 2
python main.py \
    -a RandomSampling \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m Bert \
    --seed 1127 \
    -t 5 \
    -g 2

# Margin
## 7/13 exp1 waldweihe 3
## seed 4666
## 7/15 exp2 waldweihe 3
python main.py \
    -a MarginSampling \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m Bert \
    --seed 1127 \
    -t 5 \
    -g 3

# LC
## 7/13 exp1 strauss 1
## iter. = 3 # wrong =.=
## seed 4666
## 7/15 exp2 strauss 0
python main.py \
    -a LeastConfidence \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m Bert \
    --seed 1127 \
    -t 5 \
    -g 0

# Entropy
## 7/13 exp1 strauss 8
## iter. = 3 # wrong =.=
## seed 4666
## 7/15 exp2 strauss 1
python main.py \
    -a EntropySampling \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m Bert \
    --seed 1127 \
    -t 5 \
    -g 1

# MarginDropout
## 7/15 exp2 strauss 2
python main.py \
    -a MarginSamplingDropout \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m Bert \
    --seed 1127 \
    -t 5 \
    -g 2

# LCDropout
## 7/15 exp2 strauss 3
python main.py \
    -a LeastConfidenceDropout \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m Bert \
    --seed 1127 \
    -t 5 \
    -g 3

# EntropyDropout
## 7/15 exp2 strauss 0
python main.py \
    -a EntropySamplingDropout \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m Bert \
    --seed 1127 \
    -t 5 \
    -g 0

# KMeans
## 7/15 exp2 waldweihe 2
## batch size in dataloader are 16
python main.py \
    -a KMeansSampling \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m Bert \
    --seed 1127 \
    -t 5 \
    -g 1

# KCenterGreedy
## 7/15 exp2 strauss 1
python main.py \
    -a KCenterGreedy \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m Bert \
    --seed 1127 \
    -t 5 \
    -g 1

# Bayesian
## 7/15 exp2 strauss 4
python main.py \
    -a BALDDropout \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m Bert \
    --seed 1127 \
    -t 5 \
    -g 4

# MeanSTD
## 7/15 exp2 strauss 5
python main.py \
    -a MeanSTD \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m Bert \
    --seed 1127 \
    -t 5 \
    -g 5

# BADGE
## 7/15 exp2 strauss 6
python main.py \
    -a BadgeSampling \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m Bert \
    --seed 1127 \
    -t 5 \
    -g 7

# LPL
python main.py \
    -a LossPredictionLoss \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m Bert \
    --seed 1127 \
    -t 5 \
    -g 2

# CEAL
python main.py \
    -a CEALSampling \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m Bert \
    --seed 1127 \
    -t 5 \
    -g 2