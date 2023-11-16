#!/bin/bash

python main.py \
    --exp_id 527a \
    -a RandomSampling \
    -q 200 \
    -b 50 \
    -d NaturalQuestions \
    -u True \
    -r True \
    -g 4

python main.py \
    --exp_id 527b \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d NaturalQuestions \
    -u True \
    -r True \
    -g 4

python main.py \
    --exp_id 527e \
    -a MarginSamplingDropout \
    -q 200 \
    -b 50 \
    -d NaturalQuestions \
    -u True \
    -r True \
    -g 4

python main.py \
    --exp_id 527j \
    -a MeanSTD \
    -q 200 \
    -b 50 \
    -d NaturalQuestions \
    -u True \
    -r True \
    -g 4

python main.py \
    --exp_id 527h \
    -a KMeansSampling \
    -q 200 \
    -b 50 \
    -d NaturalQuestions \
    -u True \
    -r True \
    -g 4

python main.py \
    --exp_id 527c \
    -a LeastConfidence \
    -q 200 \
    -b 50 \
    -d NaturalQuestions \
    -u True \
    -r True \
    -g 4

python main.py \
    --exp_id 527f \
    -a LeastConfidenceDropout \
    -q 200 \
    -b 50 \
    -d NaturalQuestions \
    -u True \
    -r True \
    -g 4

python main.py \
    --exp_id 527k \
    -a BALDDropout \
    -q 200 \
    -b 50 \
    -d NaturalQuestions \
    -u True \
    -r True \
    -g 4

python main.py \
    --exp_id 527i \
    -a KCenterGreedy \
    -q 200 \
    -b 50 \
    -d NaturalQuestions \
    -u True \
    -r True \
    -g 4

python main.py \
    --exp_id 527d \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d NaturalQuestions \
    -u True \
    -r True \
    -g 4

python main.py \
    --exp_id 527g \
    -a EntropySamplingDropout \
    -q 200 \
    -b 50 \
    -d NaturalQuestions \
    -u True \
    -r True \
    -g 4

python main.py \
    --exp_id 527l \
    -a BadgeSampling \
    -q 200 \
    -b 50 \
    -d NaturalQuestions \
    -u True \
    -r True \
    -g 4