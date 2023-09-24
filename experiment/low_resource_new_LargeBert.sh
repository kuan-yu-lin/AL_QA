# ####################
# AL: Margin, LC, Entropy, MarginDropout, LCDroupout, EntropyDroupout, Random, KMeans, KCenter, BALD, MeanSTD, BADGE
#
# Model: BERT Large (lr=3e-5) (only kapweihe works)
#
# source domain: SQuAD
#
# target domain: BioASQ, SearchQA(with 1500)
#
# Query: 50, 100, 150, 200
# Query batch: 50 
# Experiment Iteration: 5
# ####################

# Margin
python main_lowRes_new.py \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m BERTLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 4

## 9/22 kapweihe
python main_lowRes_new.py \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m BERTLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 4
####################################
# LC
python main_lowRes_new.py \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m BERTLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

## 9/22 kapweihe
python main_lowRes_new.py \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m BERTLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

#################################
# Entropy
python main_lowRes_new.py \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m BERTLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 5

## 9/22 kapweihe
python main_lowRes_new.py \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m BERTLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 4

####################################
# MarginDropout
python main_lowRes_new.py \
    -a MarginSamplingDropout \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m BERTLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 7

# 9/22
python main_lowRes_new.py \
    -a MarginSamplingDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m BERTLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

###################################
# LCDropout
python main_lowRes_new.py \
    -a LeastConfidenceDropout \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m BERTLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 7

# 9/22
python main_lowRes_new.py \
    -a LeastConfidenceDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m BERTLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 4

####################################
# EntropyDropout
python main_lowRes_new.py \
    -a EntropySamplingDropout \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m BERTLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 6

# 9/22
python main_lowRes_new.py \
    -a EntropySamplingDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m BERTLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

####################################
# Random
python main_lowRes_new.py \
    -a RandomSampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m BERTLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 6

## 9/22 strauss
python main_lowRes_new.py \
    -a RandomSampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m BERTLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

#################################
# KMeans
python main_lowRes_new.py \
    -a KMeansSampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m BERTLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 6

# 9/22
python main_lowRes_new.py \
    -a KMeansSampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m BERTLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

###############################
# KCenterGreedy
python main_lowRes_new.py \
    -a KCenterGreedy \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m BERTLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

# 9/22
python main_lowRes_new.py \
    -a KCenterGreedy \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m BERTLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

#################################
# Bayesian/BALD
python main_lowRes_new.py \
    -a BALDDropout \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m BERTLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 7

# 9/22
python main_lowRes_new.py \
    -a BALDDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m BERTLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 4

##################################
# MeanSTD
python main_lowRes_new.py \
    -a MeanSTD \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m BERTLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 8

# 9/23
python main_lowRes_new.py \
    -a MeanSTD \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m BERTLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

##################################
# BADGE
python main_lowRes_new.py \
    -a BadgeSampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m BERTLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 6

# 9/22
python main_lowRes_new.py \
    -a BadgeSampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m BERTLarge \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 4
