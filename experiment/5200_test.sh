#!/bin/bash

python main.py \
    --exp_id 123ht \
    --exp_round 1 \
    -a KMeansSampling \
    -q 200 \
    -b 50 \
    -d DROP \
    -r True \
    -g 3

python main.py \
    --exp_id 522at \
	--exp_round 1 \
    -a RandomSampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 2

python main.py \
    --exp_id 522bt \
	--exp_round 1 \
    -a MarginSampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 2

# python main.py \
#     --exp_id 522et \
# 	--exp_round 1 \
#     -a MarginSamplingDropout \
#     -q 200 \
#     -b 50 \
#     -d BioASQ \
#     -u True \
#     -r True \
#     -g 2

python main.py \
    --exp_id 522jt \
	--exp_round 1 \
    -a MeanSTD \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 2

# python main.py \
#     --exp_id 523ht \
# 	--exp_round 1 \
#     -a KMeansSampling \
#     -q 200 \
#     -b 50 \
#     -d DROP \
#     -u True \
#     -r True \
#     -g 2

# python main.py \
#     --exp_id 523ct \
#     --exp_round 1 \
#     -a LeastConfidence \
#     -q 200 \
#     -b 50 \
#     -d DROP \
#     -u True \
#     -r True \
#     -g 2

# python main.py \
#     --exp_id 522f \
# 	--exp_round 1 \
#     -a LeastConfidenceDropout \
#     -q 200 \
#     -b 50 \
#     -d BioASQ \
#     -u True \
#     -r True \
#     -g 2

python main.py \
    --exp_id 522kt \
	--exp_round 1 \
    -a BALDDropout \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 2

python main.py \
    --exp_id 522it \
	--exp_round 1 \
    -a KCenterGreedy \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 2

python main.py \
    --exp_id 522dt \
	--exp_round 1 \
    -a EntropySampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 2

# python main.py \
#     --exp_id 522g \
# 	--exp_round 1 \
#     -a EntropySamplingDropout \
#     -q 200 \
#     -b 50 \
#     -d BioASQ \
#     -u True \
#     -r True \
#     -g 2

python main.py \
    --exp_id 522lt \
	--exp_round 1 \
    -a BadgeSampling \
    -q 200 \
    -b 50 \
    -d BioASQ \
    -u True \
    -r True \
    -g 2