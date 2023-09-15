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
    -l 1e-4 \
    -g 3

# Random
## 9/3 strauss
## 9/12 strauss
## 9/13 strauss (test lr=1e-4)
python main_lowRes.py \
    -a RandomSampling \
    -q 200 \
    -b 50 \
    -d DROP \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 1e-4 \
    -p 5 \
    -g 2

## 9/12 strauss
## 9/13 strauss (test lr=1e-4)
python main_lowRes.py \
    -a RandomSampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 1e-4 \
    -p 5 \
    -g 6

## 9/12 strauss
## 9/13 strauss (test lr=1e-4)
python main_lowRes.py \
    -a RandomSampling \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 1e-4 \
    -p 5 \
    -g 3

## 9/12 strauss
## 9/13 strauss (test lr=1e-4)
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
## 9/13 strauss (test lr=1e-4)
python main_lowRes.py \
    -a RandomSampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 1e-4 \
    -p 5 \
    -g 4

######################################
# Margin
## 9/13 strauss (fail)
## 9/13 strauss
## 9/14 strauss (test lr=1e-4)
python main_lowRes.py \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d DROP \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 1e-4 \
    -p 5 \
    -g 3

## 9/13 kapweihe
## 9/14 strauss (test lr=1e-4)
python main_lowRes.py \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 1e-4 \
    -p 5 \
    -g 4

python main_lowRes.py \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 8
 
## 9/13 strauss (fail)
python main_lowRes.py \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 6

## 9/13 kapweihe (fail)
python main_lowRes.py \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

####################################
# LC
## 9/13 strauss
## 9/14 strauss (test lr=1e-4)
python main_lowRes.py \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d DROP \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 1e-4 \
    -p 5 \
    -g 2

## 9/13 strauss
## 9/14 strauss (test lr=1e-4)
python main_lowRes.py \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 1e-4 \
    -p 5 \
    -g 6

python main_lowRes.py \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 6

python main_lowRes.py \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 4

## 9/13 strauss
python main_lowRes.py \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

#################################
# Entropy
## 7/13 exp1 strauss 8
## iter. = 3 # wrong =.=
## 9/13 strauss
## 9/14 strauss (test lr=1e-4)
python main_lowRes.py \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d DROP \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 1e-4 \
    -p 5 \
    -g 3

## 9/13 strauss
## 9/14 strauss (test lr=1e-4)
python main_lowRes.py \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

python main_lowRes.py \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 4

python main_lowRes.py \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 6

python main_lowRes.py \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 6

####################################
# MarginDropout
python main_lowRes.py \
    -a MarginSamplingDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

# LCDropout
python main_lowRes.py \
    -a LeastConfidenceDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 2

# EntropyDropout
python main_lowRes.py \
    -a EntropySamplingDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 2

####################################
# KMeans
## 9/14 strauss
## 9/14 strauss (test lr=1e-4)
python main_lowRes.py \
    -a KMeansSampling \
    -q 200 \
    -b 50 \
    -d DROP \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 1e-4 \
    -p 5 \
    -g 6

## 9/14 strauss
## 9/14 strauss (test lr=1e-4)
python main_lowRes.py \
    -a KMeansSampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 1e-4 \
    -p 5 \
    -g 6

###############################
# KCenterGreedy
python main_lowRes.py \
    -a KCenterGreedy \
    -q 200 \
    -b 50 \
    -d DROP \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 4

python main_lowRes.py \
    -a KCenterGreedy \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 6

#################################
# Bayesian
python main_lowRes.py \
    -a BALDDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

# MeanSTD
python main_lowRes.py \
    -a MeanSTD \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 2

# BADGE
python main_lowRes.py \
    -a BadgeSampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 0

# LPL
python main_lowRes.py \
    -a LossPredictionLoss \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 2

# CEAL
python main_lowRes.py \
    -a CEALSampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 2