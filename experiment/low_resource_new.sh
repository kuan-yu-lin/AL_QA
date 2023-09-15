# ####################
# Model: RoBERTa-base (lr=3e-5)
#
# source domain: SQuAD
#
# target domain: BioASQ, TextbookQA, DROP, NewsQA, SearchQA
#
# Query: 50, 100, 150, 200
# Query batch: 50 
# Experiment Iteration: 5
# ####################

# Random
## 9/15 strauss
python main_lowRes_new.py \
    -a RandomSampling \
    -q 200 \
    -b 50 \
    -d DROP \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

## 9/15 strauss
python main_lowRes_new.py \
    -a RandomSampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 4

## 9/15 strauss
python main_lowRes_new.py \
    -a RandomSampling \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 5

## 9/15 strauss
python main_lowRes_new.py \
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

## 9/15 strauss
python main_lowRes_new.py \
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

######################################
# Margin
## 9/15 strauss
python main_lowRes_new.py \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d DROP \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 4

## 9/15 strauss
python main_lowRes_new.py \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 5

## 9/15 strauss
python main_lowRes_new.py \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 4

## 9/15 strauss
python main_lowRes_new.py \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 5

## 9/15 strauss
python main_lowRes_new.py \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 4

####################################
# LC
## 9/15 strauss (done in comparison)
python main_lowRes_new.py \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d DROP \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 6

## 9/15 strauss (done in comparison)
python main_lowRes_new.py \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 6

## 9/15 strauss
python main_lowRes_new.py \
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

## 9/15 strauss
python main_lowRes_new.py \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 6

###### hold-up
python main_lowRes_new.py \
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
## 9/15 strauss
python main_lowRes_new.py \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d DROP \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

## 9/15 strauss
python main_lowRes_new.py \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 2

## 9/15 strauss
python main_lowRes_new.py \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

python main_lowRes_new.py \
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

python main_lowRes_new.py \
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
python main_lowRes_new.py \
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
python main_lowRes_new.py \
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
## 9/15 strauss
python main_lowRes_new.py \
    -a EntropySamplingDropout \
    -q 200 \
    -b 50 \
    -d DROP \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

####################################
# KMeans
## 9/15 strauss (done in comparison)
python main_lowRes_new.py \
    -a KMeansSampling \
    -q 200 \
    -b 50 \
    -d DROP \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 6

## 9/15 strauss (done in comparison)
python main_lowRes_new.py \
    -a KMeansSampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 6

###############################
# KCenterGreedy
python main_lowRes_new.py \
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

python main_lowRes_new.py \
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
python main_lowRes_new.py \
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
python main_lowRes_new.py \
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
python main_lowRes_new.py \
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
python main_lowRes_new.py \
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
python main_lowRes_new.py \
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