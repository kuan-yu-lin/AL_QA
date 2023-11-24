#!/bin/bash

python main.py \
	--exp_id 111at \
	--exp_round 1 \
	--dev_mode True \
    -a RandomSampling \
    -g 7

python main.py \
    --exp_id 123ht \
    --exp_round 1 \
    -q 200 \
    -b 50 \
    -g 7