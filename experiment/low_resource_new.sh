# ####################
# Part 1: Margin, LC, Entropy, MarginDropout, LCDroupout, EntropyDroupout
#
# Model: RoBERTa-base (lr=3e-5)
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
## 9/17 strauss (with 1500 data)
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
    -g 6

## 9/15 strauss
## 9/17 strauss (with 1500 data)
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
    -g 3

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
## 9/17 strauss (with 1500 data)
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
    -g 7

###### hold-up
## 9/17 strauss (with 1500 data)
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
    -g 5

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

###### hold-up
## 9/17 strauss (with 1500 data)
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
    -g 8

###### hold-up
## 9/17 strauss (with 1500 data)
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
    -g 2

####################################
# MarginDropout
## 9/16 strauss
python main_lowRes_new.py \
    -a MarginSamplingDropout \
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
    -a MarginSamplingDropout \
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
    -a MarginSamplingDropout \
    -q 200 \
    -b 50 \
    -d TextbookQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 6

###### hold-up
## 9/17 strauss (with 1500 data)
python main_lowRes_new.py \
    -a MarginSamplingDropout \
    -q 200 \
    -b 50 \
    -d NewsQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

###### hold-up
## 9/17 strauss (with 1500 data)
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
    -g 5

###################################
# LCDropout
## 9/16 strauss
python main_lowRes_new.py \
    -a LeastConfidenceDropout \
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
    -a LeastConfidenceDropout \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 3

## 9/16 strauss
python main_lowRes_new.py \
    -a LeastConfidenceDropout \
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
    -a LeastConfidenceDropout \
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
    -a LeastConfidenceDropout \
    -q 200 \
    -b 50 \
    -d SearchQA \
    -m RoBERTa \
    -r True \
    -e 1127 \
    -l 3e-5 \
    -p 5 \
    -g 7

####################################
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

## 9/16 strauss
python main_lowRes_new.py \
    -a EntropySamplingDropout \
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
    -a EntropySamplingDropout \
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
    -a EntropySamplingDropout \
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
