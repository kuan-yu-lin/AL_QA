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
    -m RoBERTaLarge \
    -e 1127 \
    -l 3e-5 \
    -g 3

python pretrain.py \
    -d SQuAD \
    -m BERTLarge \
    -e 1127 \
    -l 3e-5 \
    -g 4

python pretrain.py \
    -d SQuAD \
    -m RoBERTa \
    -e 1127 \
    -l 3e-5 \
    -g 3

python pretrain.py \
    -d SQuAD \
    -m Bert \
    -e 1127 \
    -l 3e-5 \
    -g 4
