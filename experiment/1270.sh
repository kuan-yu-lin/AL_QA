#!/bin/bash

python main.py \
    --exp_id 127a \
    -a RandomSampling \
    -q 200 \
    -b 50 \
    -d NaturalQuestions \
    -r True \
    -g 3

python main.py \
    --exp_id 127b \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d NaturalQuestions \
    -r True \
    -g 3

python main.py \
    --exp_id 127c \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d NaturalQuestions \
    -r True \
    -g 3

python main.py \
    --exp_id 127d \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d NaturalQuestions \
    -r True \
    -g 3

python main.py \
    --exp_id 127e \
    -a MarginSamplingDropout \
    -q 200 \
    -b 50 \
    -d NaturalQuestions \
    -r True \
    -g 3

python main.py \
    --exp_id 127f \
    -a LeastConfidenceDropout \
    -q 200 \
    -b 50 \
    -d NaturalQuestions \
    -r True \
    -g 3

python main.py \
    --exp_id 127g \
    -a EntropySamplingDropout \
    -q 200 \
    -b 50 \
    -d NaturalQuestions \
    -r True \
    -g 3

python main.py \
    --exp_id 127h \
    -a KMeansSampling \
    -q 200 \
    -b 50 \
    -d NaturalQuestions \
    -r True \
    -g 3

python main.py \
    --exp_id 127i \
    -a KCenterGreedy \
    -q 200 \
    -b 50 \
    -d NaturalQuestions \
    -r True \
    -g 3

python main.py \
    --exp_id 127j \
    -a MeanSTD \
    -q 200 \
    -b 50 \
    -d NaturalQuestions \
    -r True \
    -g 3

python main.py \
    --exp_id 127k \
    -a BALDDropout \
    -q 200 \
    -b 50 \
    -d NaturalQuestions \
    -r True \
    -g 3

python main.py \
    --exp_id 127l \
    -a BadgeSampling \
    -q 200 \
    -b 50 \
    -d NaturalQuestions \
    -r True \
    -g 3
