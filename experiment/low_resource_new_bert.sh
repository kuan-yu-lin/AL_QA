# ####################
# Part 1: Margin, LC, Entropy, MarginDropout, LCDroupout, EntropyDroupout
#
# Model: Bert-base (lr=3e-5) (fixed)
#
# source domain: SQuAD
#
# target domain: DROP, BioASQ, TextbookQA, NewsQA, SearchQA
#
# Query: 50, 100, 150, 200
# Query batch: 50 
# Experiment Iteration: 5
# ####################

# Margin
## 9/18 strauss
## 9/22 strauss
python main_lowRes_new.py \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d DROP \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 4

## 9/19 strauss
## 9/22 kapweihe
python main_lowRes_new.py \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 4

## 9/19 strauss
## 9/22 kapweihe
python main_lowRes_new.py \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

## 9/19 strauss (with 1500 data)
## 9/21 strauss
python main_lowRes_new.py \
    -a MarginSampling \
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
## 9/21 kapweihe
python main_lowRes_new.py \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

####################################
# LC
## 9/18 strauss
## 9/21 kapweihe
python main_lowRes_new.py \
    -a LeastConfidence \
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
    -a LeastConfidence \
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
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

## 9/19 strauss (with 1500 data)
## 9/21 kapweihe
python main_lowRes_new.py \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

## 9/19 strauss (with 1500 data)
## 9/21 kapweihe
python main_lowRes_new.py \
    -a LeastConfidence \
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
# Entropy
## 9/18 strauss
## 9/21 strauss
python main_lowRes_new.py \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d DROP \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 6

## 9/19 strauss
## 9/21 strauss
python main_lowRes_new.py \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 5

## 9/19 strauss
## 9/21 strauss
python main_lowRes_new.py \
    -a EntropySampling \
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
    -a EntropySampling \
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
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 8

####################################
# MarginDropout
## 9/18 strauss
## 9/21 strauss
python main_lowRes_new.py \
    -a MarginSamplingDropout \
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
    -a MarginSamplingDropout \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 7

## 9/19 strauss
## 9/21 strauss
python main_lowRes_new.py \
    -a MarginSamplingDropout \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 6

## 9/19 strauss (with 1500 data)
## 9/21 strauss
python main_lowRes_new.py \
    -a MarginSamplingDropout \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 5

## 9/20 strauss (with 1500 data)
## 9/21 strauss
python main_lowRes_new.py \
    -a MarginSamplingDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 4

###################################
# LCDropout
## 9/18 strauss
## 9/21 strauss
python main_lowRes_new.py \
    -a LeastConfidenceDropout \
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
    -a LeastConfidenceDropout \
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
## 9/21 strauss
python main_lowRes_new.py \
    -a LeastConfidenceDropout \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 6

## 9/20 strauss (with 1500 data)
## 9/21 strauss
python main_lowRes_new.py \
    -a LeastConfidenceDropout \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 5

## 9/20 strauss (with 1500 data)
## 9/21 strauss
python main_lowRes_new.py \
    -a LeastConfidenceDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 4

####################################
# EntropyDropout
## 9/18 strauss
## 9/21 strauss
python main_lowRes_new.py \
    -a EntropySamplingDropout \
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
## 9/20 strauss
python main_lowRes_new.py \
    -a EntropySamplingDropout \
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
## 9/20 strauss
python main_lowRes_new.py \
    -a EntropySamplingDropout \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 7

## 9/20 strauss (with 1500 data)

python main_lowRes_new.py \
    -a EntropySamplingDropout \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 2

## 9/20 strauss (with 1500 data)

python main_lowRes_new.py \
    -a EntropySamplingDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m Bert \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 4
