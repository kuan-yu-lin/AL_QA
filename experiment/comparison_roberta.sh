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
## 7/17 (archived)
## 9/3 strauss (archived)
## 9/7 kapweihe
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
    -g 3

# LC
## 7/18 (archived)
## 9/3 strauss (archived)
## 9/7 strauss
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
    -g 8

# Entropy
## 7/18 (archived)
## 9/3 waldweihe (test tmux)
## 9/3 strass (archived)
## 9/7 kapweihe
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
    -g 3

# MarginDropout
## 9/4 strass (not finished)
## 9/6 strass
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
## 9/6 strass
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
## 9/6 strass
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
## 9/12 strass 
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
    -g 3

# KCenterGreedy
## 7/17
## 9/12 strass
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
    -g 4

# BALD
## 9/12 strass (fail because of accidently delete the saved model = =")
## 9/14 kapweihe 
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
## 9/6 strauss
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
    -g 3

# BADGE
## 7/17-? kapweihe
## 9/6 kapweihe
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
    -g 4

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