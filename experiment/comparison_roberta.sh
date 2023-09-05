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
## 9/3 strauss
## 9/6 kapweihe
python main.py \
    -a RandomSampling \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

# Margin
## 7/17
## 9/3 strauss
python main.py \
    -a MarginSampling \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 6

# LC
## 7/18
## 9/3 strauss
python main.py \
    -a LeastConfidence \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 4

# Entropy
## 7/18
## 9/3 waldweihe (test tmux)
## 9/3 strass
python main.py \
    -a EntropySampling \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 4

# MarginDropout
## 9/4 strass
python main.py \
    -a MarginSamplingDropout \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 4

# LCDropout
## 7/17-? kapweihe
## 9/3 strass
python main.py \
    -a LeastConfidenceDropout \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 5

# EntropyDropout
## 9/4 strass
python main.py \
    -a EntropySamplingDropout \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 6

# KMeans
## 9/3 waldweihe
python main.py \
    -a KMeansSampling \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 1

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
    -p 5 \
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
    -p 5 \
    -g 3

# MeanSTD
## 9/4 kapweihe
python main.py \
    -a MeanSTD \
    -s 500 \
    -q 2000 \
    -b 500 \
    -d SQuAD \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 4

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
    -p 5 \
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
#     -p 5 \
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
#     -p 5 \
#     -g 2