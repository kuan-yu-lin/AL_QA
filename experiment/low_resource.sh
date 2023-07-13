# ####################
# Model: Roberta-base
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

# Random
## 7/13 exp1 waldweihe 2
python main_lowRes.py \
    -a RandomSampling \
    -s 50 \
    -q 200 \
    -b 50 \
    -d SQuAD \
    -m RoBERTa \
    --seed 1127 \
    -t 5 \
    -g 2

# Margin
## 7/13 exp1 waldweihe 3
python main_lowRes.py \
    -a MarginSampling \
    -s 50 \
    -q 200 \
    -b 50 \
    -d SQuAD \
    -m RoBERTa \
    --seed 1127 \
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
    -d SQuAD \
    -m RoBERTa \
    --seed 1127 \
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
    -d SQuAD \
    -m RoBERTa \
    --seed 1127 \
    -t 5 \
    -g 8

# MarginDropout
python main_lowRes.py \
    -a MarginSamplingDropout \
    -s 100 \
    -q 100 \
    -b 35 \
    -d SQuAD \
    -m RoBERTa \
    --seed 1127 \
    -t 1 \
    -g 3

# LCDropout
python main_lowRes.py \
    -a LeastConfidenceDropout \
    -s 100 \
    -q 100 \
    -b 35 \
    -d SQuAD \
    -m RoBERTa \
    --seed 1127 \
    -t 1 \
    -g 2

# EntropyDropout
python main_lowRes.py \
    -a EntropySamplingDropout \
    -s 100 \
    -q 100 \
    -b 35 \
    -d SQuAD \
    -m RoBERTa \
    --seed 1127 \
    -t 1 \
    -g 2

# KMeans
python main_lowRes.py \
    -a KMeansSampling \
    -s 2000 \
    -q 10000 \
    -b 3000 \
    -d SQuAD \
    -m RoBERTa \
    --seed 1127 \
    -t 5 \
    -g 3

# KCenterGreedy
python main_lowRes.py \
    -a KCenterGreedy \
    -s 100 \
    -q 100 \
    -b 35 \
    -d SQuAD \
    -m RoBERTa \
    --seed 1127 \
    -t 1 \
    -g 2

# Bayesian
python main_lowRes.py \
    -a BALDDropout \
    -s 2000 \
    -q 10000 \
    -b 3000 \
    -d SQuAD \
    -m RoBERTa \
    --seed 1127 \
    -t 5 \
    -g 3

# MeanSTD
python main_lowRes.py \
    -a MeanSTD \
    -s 100 \
    -q 100 \
    -b 35 \
    -d SQuAD \
    -m RoBERTa \
    --seed 1127 \
    -t 1 \
    -g 2

# BADGE
python main_lowRes.py \
    -a BadgeSampling \
    -s 2000 \
    -q 10000 \
    -b 3000 \
    -d SQuAD \
    -m RoBERTa \
    --seed 1127 \
    -t 5 \
    -g 0

# LPL
python main_lowRes.py \
    -a LossPredictionLoss \
    -s 100 \
    -q 100 \
    -b 35 \
    -d SQuAD \
    -m RoBERTa \
    --seed 1127 \
    -t 1 \
    -g 2

# CEAL
python main_lowRes.py \
    -a CEALSampling \
    -s 100 \
    -q 100 \
    -b 35 \
    -d SQuAD \
    -m RoBERTa \
    --seed 1127 \
    -t 1 \
    -g 2