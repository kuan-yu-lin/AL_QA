#!/bin/bash

# python main.py \
# 	--exp_id 111at \
# 	--exp_round 1 \
# 	--dev_mode True \
#     -g 2

python main.py \
	--exp_id 111ct \
	--exp_round 1 \
	--dev_mode True \
	-g 5

python main_adapter.py \
    --exp_id 122at \
    --exp_round 1 \
    -q 200 \
    -b 50 \
    -g 3

python main_adapter.py \
    --exp_id 122bt \
    --exp_round 1 \
    -q 200 \
    -b 50 \
    -g 3

python main_adapter.py \
    --exp_id 122ct \
    --exp_round 1 \
    -q 200 \
    -b 50 \
    -g 3

python main_adapter.py \
    --exp_id 122dt \
    --exp_round 1 \
    -q 200 \
    -b 50 \
    -g 3

python main_adapter.py \
    --exp_id 122et \
    --exp_round 1 \
    -q 200 \
    -b 50 \
    -g 3

python main_adapter.py \
    --exp_id 122ft \
    --exp_round 1 \
    -q 200 \
    -b 50 \
    -g 3

python main_adapter.py \
    --exp_id 122gt \
    --exp_round 1 \
    -q 200 \
    -b 50 \
    -g 3

python main_adapter.py \
    --exp_id 122ht \
    --exp_round 1 \
    -q 200 \
    -b 50 \
    -g 3

python main_adapter.py \
    --exp_id 122it \
    --exp_round 1 \
    -q 200 \
    -b 50 \
    -g 3

python main_adapter.py \
    --exp_id 122jt \
    --exp_round 1 \
    -q 200 \
    -b 50 \
    -g 3

python main_adapter.py \
    --exp_id 122kt \
    --exp_round 1 \
    -q 200 \
    -b 50 \
    -g 3

python main_adapter.py \
    --exp_id 122lt \
    --exp_round 1 \
    -q 200 \
    -b 50 \
    -g 3