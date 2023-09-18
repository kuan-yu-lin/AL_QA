# ####################
# Part 2: Random, KMeans, KCenter, BALD, MeanSTD, BADGE
#
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
## 9/17 strauss (with 1500 data)
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
    -g 3

## 9/15 strauss
## 9/17 strauss (with 1500 data)
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
    -g 6

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

## 9/16 strauss
python main_lowRes_new.py \
    -a KMeansSampling \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 7

## 9/17 strauss (with 1500 data)
python main_lowRes_new.py \
    -a KMeansSampling \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

## 9/17 strauss (with 1500 data)
python main_lowRes_new.py \
    -a KMeansSampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 5

###############################
# KCenterGreedy
## 9/16 strauss
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
    -g 3

## 9/16 strauss
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
    -g 8

## 9/16 strauss
python main_lowRes_new.py \
    -a KCenterGreedy \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

## 9/17 strauss (with 1500 data)
python main_lowRes_new.py \
    -a KCenterGreedy \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 6

## 9/17 strauss (with 1500 data)
python main_lowRes_new.py \
    -a KCenterGreedy \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 7

#################################
# Bayesian
## 9/16 strauss
python main_lowRes_new.py \
    -a BALDDropout \
    -q 200 \
    -b 50 \
    -d DROP \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

## 9/16 strauss
python main_lowRes_new.py \
    -a BALDDropout \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 5

## 9/16 strauss
python main_lowRes_new.py \
    -a BALDDropout \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 8

## 9/17 strauss (with 1500 data)
python main_lowRes_new.py \
    -a BALDDropout \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 8

## 9/17 strauss (with 1500 data)
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
    -g 2

##################################
# MeanSTD
## 9/16 strauss
python main_lowRes_new.py \
    -a MeanSTD \
    -q 200 \
    -b 50 \
    -d DROP \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 8

## 9/16 strauss
python main_lowRes_new.py \
    -a MeanSTD \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 7

## 9/16 strauss
python main_lowRes_new.py \
    -a MeanSTD \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

## 9/17 strauss (with 1500 data)
python main_lowRes_new.py \
    -a MeanSTD \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 7

## 9/17 strauss (with 1500 data)
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
    -g 5

##################################
# BADGE
## 9/16 strauss
python main_lowRes_new.py \
    -a BadgeSampling \
    -q 200 \
    -b 50 \
    -d DROP \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 8

## 9/16 strauss
python main_lowRes_new.py \
    -a BadgeSampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 7

## 9/16 strauss
python main_lowRes_new.py \
    -a BadgeSampling \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 5

## 9/17 strauss (with 1500 data)
python main_lowRes_new.py \
    -a BadgeSampling \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

## 9/17 strauss (with 1500 data)
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
    -g 6
