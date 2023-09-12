# ####################
# Model: RoBERTa-base
#
# source domain: SQuAD
#
# target domain: BioASQ, TextbookQA, DROP, NewsQA, SearchQA
#
# Query: 50, 100, 150, 200
# Query batch: 50 
# Experiment Iteration: 5
# ####################

# Pretrain
python pretrain.py \
    -d SQuAD \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -g 5

# Random
## 9/3 strauss
## 9/12 strauss
python main_lowRes.py \
    -a RandomSampling \
    -q 200 \
    -b 50 \
    -d DROP \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 6

## 9/12 strauss
python main_lowRes.py \
    -a RandomSampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 6

python main_lowRes.py \
    -a RandomSampling \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 8

python main_lowRes.py \
    -a RandomSampling \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 6

## 9/12 kapweihe
python main_lowRes.py \
    -a RandomSampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

# Margin
python main_lowRes.py \
    -a MarginSampling \
    -s 50 \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 8

# LC
python main_lowRes.py \
    -a LeastConfidence \
    -s 50 \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 1

# Entropy
## 7/13 exp1 strauss 8
## iter. = 3 # wrong =.=
python main_lowRes.py \
    -a EntropySampling \
    -s 50 \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 8

# MarginDropout
python main_lowRes.py \
    -a MarginSamplingDropout \
    -s 50 \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

# LCDropout
python main_lowRes.py \
    -a LeastConfidenceDropout \
    -s 50 \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 2

# EntropyDropout
python main_lowRes.py \
    -a EntropySamplingDropout \
    -s 50 \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 2

# KMeans
python main_lowRes.py \
    -a KMeansSampling \
    -s 50 \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

# KCenterGreedy
python main_lowRes.py \
    -a KCenterGreedy \
    -s 50 \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 2

# Bayesian
python main_lowRes.py \
    -a BALDDropout \
    -s 50 \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

# MeanSTD
python main_lowRes.py \
    -a MeanSTD \
    -s 50 \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 2

# BADGE
python main_lowRes.py \
    -a BadgeSampling \
    -s 50 \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 0

# LPL
python main_lowRes.py \
    -a LossPredictionLoss \
    -s 50 \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 2

# CEAL
python main_lowRes.py \
    -a CEALSampling \
    -s 50 \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 2