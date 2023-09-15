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

# LC
## 9/14 kapweihe
## 9/14 strauss (test lr=1e-4)
## 9/14 strauss (pre-t model lr=1e-4)
## 9/15 strauss (pre-t model lr=1e-4) (test lr=1e-4)
## 9/15 strauss (_new)(pre-t model lr=1e-4) (test lr=1e-4)
## 9/15 strauss (_new)(pre-t model lr=1e-4)
## 9/15 strauss (_new)
## 9/15 strauss (_new)(test lr=1e-4)

python main_lowRes_new.py \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d DROP \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 1e-4 \
    -p 5 \
    -g 5

## 9/13 strauss
## 9/14 strauss (test lr=1e-4)
## 9/14 strauss (pre-t model lr=1e-4)
## 9/15 strauss (pre-t model lr=1e-4) (test lr=1e-4)
## 9/15 strauss (_new)(pre-t model lr=1e-4) (test lr=1e-4)
## 9/15 strauss (_new)(pre-t model lr=1e-4)
## 9/15 strauss (_new)
## 9/15 strauss (_new)(test lr=1e-4)

python main_lowRes_new.py \
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

####################################
# KMeans
## 9/14 strauss
## 9/14 strauss (test lr=1e-4)
## 9/14 strauss (pre-t model lr=1e-4)
## 9/15 strauss (pre-t model lr=1e-4) (test lr=1e-4)
## 9/15 strauss (_new)(pre-t model lr=1e-4) (test lr=1e-4)
## 9/15 strauss (_new)(pre-t model lr=1e-4)
## 9/15 strauss (_new)
## 9/15 strauss (_new)(test lr=1e-4)


python main_lowRes_new.py \
    -a KMeansSampling \
    -q 200 \
    -b 50 \
    -d DROP \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 1e-4 \
    -p 5 \
    -g 3

## 9/14 strauss
## 9/14 strauss (test lr=1e-4)
## 9/14 strauss (pre-t model lr=1e-4)
## 9/15 strauss (pre-t model lr=1e-4) (test lr=1e-4)
## 9/15 strauss (_new)(pre-t model lr=1e-4) (test lr=1e-4)
## 9/15 strauss (_new)(pre-t model lr=1e-4)
## 9/15 strauss (_new)
## 9/15 strauss (_new)(test lr=1e-4)


python main_lowRes_new.py \
    -a KMeansSampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 1e-4 \
    -p 5 \
    -g 4
