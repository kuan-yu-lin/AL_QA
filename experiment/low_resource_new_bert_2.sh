# ####################
# Part 2: Random, KMeans, KCenter, BALD, MeanSTD, BADGE
#
# Model: Bert-base (lr=3e-5) (fixed)
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
## 9/19 strauss
## 9/21 strauss (with 1500 data)
python main_lowRes_new.py \
    -a RandomSampling \
    -q 200 \
    -b 50 \
    -d DROP \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 7

## 9/19 strauss
## 9/21 strauss (with 1500 data)
python main_lowRes_new.py \
    -a RandomSampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 6

## 9/19 strauss
## 9/21 strauss (with 1500 data)
python main_lowRes_new.py \
    -a RandomSampling \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 5

## 9/19 strauss (with 1500 data)
## 9/21 strauss (with 1500 data)
python main_lowRes_new.py \
    -a RandomSampling \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 6

## 9/19 strauss (with 1500 data)
## 9/21 strauss (with 1500 data)
python main_lowRes_new.py \
    -a RandomSampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 5

####################################
# KMeans
## 9/19 strauss
## 9/21 strauss
python main_lowRes_new.py \
    -a KMeansSampling \
    -q 200 \
    -b 50 \
    -d DROP \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 5

## 9/19 strauss
## 9/21 strauss
python main_lowRes_new.py \
    -a KMeansSampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 6

## 9/19 strauss
## 9/21 strauss
python main_lowRes_new.py \
    -a KMeansSampling \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 5

## 9/19 strauss (with 1500 data)
## 9/21 kapweihe
python main_lowRes_new.py \
    -a KMeansSampling \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 4

## 9/19 strauss (with 1500 data)
## 9/21 kapweihe
python main_lowRes_new.py \
    -a KMeansSampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

###############################
# KCenterGreedy
## 9/19 strauss
## 9/21 kapweihe
python main_lowRes_new.py \
    -a KCenterGreedy \
    -q 200 \
    -b 50 \
    -d DROP \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

## 9/19 strauss
## 9/21 kapweihe
python main_lowRes_new.py \
    -a KCenterGreedy \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

## 9/19 strauss
## 9/21 kapweihe
python main_lowRes_new.py \
    -a KCenterGreedy \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 4

## 9/19 strauss (with 1500 data)
## 9/21 kapweihe
python main_lowRes_new.py \
    -a KCenterGreedy \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 4

## 9/19 strauss (with 1500 data)
## 9/21 kapweihe
python main_lowRes_new.py \
    -a KCenterGreedy \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

#################################
# Bayesian/BALD
## 9/18 strauss
## 9/21 strauss
python main_lowRes_new.py \
    -a BALDDropout \
    -q 200 \
    -b 50 \
    -d DROP \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 8

## 9/19 strauss
## 9/21 strauss
python main_lowRes_new.py \
    -a BALDDropout \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 7

## 9/20 strauss
python main_lowRes_new.py \
    -a BALDDropout \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 5

## 9/20 strauss (with 1500 data)
python main_lowRes_new.py \
    -a BALDDropout \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 6

## 9/20 strauss (with 1500 data)
python main_lowRes_new.py \
    -a BALDDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 7

##################################
# MeanSTD
## 9/18 strauss
## 9/21 strauss
python main_lowRes_new.py \
    -a MeanSTD \
    -q 200 \
    -b 50 \
    -d DROP \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 8

## 9/20 strauss
python main_lowRes_new.py \
    -a MeanSTD \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 8

## 9/20 strauss
python main_lowRes_new.py \
    -a MeanSTD \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 8

## 9/20 strauss (with 1500 data)
python main_lowRes_new.py \
    -a MeanSTD \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 4

## 9/20 strauss (with 1500 data)
python main_lowRes_new.py \
    -a MeanSTD \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 5

##################################
# BADGE
## 9/20 strauss
python main_lowRes_new.py \
    -a BadgeSampling \
    -q 200 \
    -b 50 \
    -d DROP \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 7

## 9/20 strauss
python main_lowRes_new.py \
    -a BadgeSampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 6

## 9/20 strauss
python main_lowRes_new.py \
    -a BadgeSampling \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 2

## 9/20 strauss (with 1500 data)
python main_lowRes_new.py \
    -a BadgeSampling \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 8

## 9/19 strauss (with 1500 data)
## 9/21 strauss (with 1500 data)
python main_lowRes_new.py \
    -a BadgeSampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 4
