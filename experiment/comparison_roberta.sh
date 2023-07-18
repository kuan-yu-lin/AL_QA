#####################
# how well they work
# Model: Bert-base Bert-large(how long) RoBERTa \-large RoBERTa \-base(http://arxiv.org/abs/2101.00438)
#
# Init pool: 500
# Query: 500, 1000, 1500, 2000 (mention it the proposal)
# Query batch: 500
# Experiment Iteration: 5
#####################

# Random
## 7/17
python main.py \
    -a RandomSampling \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -t 5 \
    -g 2

# Margin
## 7/17
python main.py \
    -a MarginSampling \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -t 5 \
    -g 3

# LC
## 7/18
python main.py \
    -a LeastConfidence \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -t 5 \
    -g 2

# Entropy
## 7/18
python main.py \
    -a EntropySampling \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -t 5 \
    -g 8

# MarginDropout
python main.py \
    -a MarginSamplingDropout \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -t 5 \
    -g 3

# LCDropout
## 7/17-? kapweihe
python main.py \
    -a LeastConfidenceDropout \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -t 5 \
    -g 2

# EntropyDropout
python main.py \
    -a EntropySamplingDropout \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -t 5 \
    -g 2

# KMeans
python main.py \
    -a KMeansSampling \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -t 5 \
    -g 3

# KCenterGreedy
## 7/17
python main.py \
    -a KCenterGreedy \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -t 5 \
    -g 2

# BALD
python main.py \
    -a BALDDropout \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -t 5 \
    -g 3

# MeanSTD
python main.py \
    -a MeanSTD \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -t 5 \
    -g 2

# BADGE
## 7/17-? kapweihe
python main.py \
    -a BadgeSampling \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -t 5 \
    -g 0

# LPL
# python main.py \
#     -a LossPredictionLoss \
#     -s 500 \
#     -q 2000 \
#     -b 500 \
#     -d SQuAD \
#     -m RoBERTa \
#     -e 1127 \
#     -l 3e-5 \
#     -t 5 \
#     -g 2

# CEAL
# python main.py \
#     -a CEALSampling \
#     -s 500 \
#     -q 2000 \
#     -b 500 \
#     -d SQuAD \
#     -m RoBERTa \
#     -e 1127 \
#     -l 3e-5 \
#     -t 5 \
#     -g 2