# ActiveLearning_QuestionAnswering

To-do list:
1. set up all with main_with_pretrain
2. consider to saperate the first random training (3 mins difference with init=2000 batch=8)

## pretrain
```bash
  python pretrain.py \
      -s 100 \
      -d SQuAD \
      --seed 4666 \
      -g 2
```
## run
```bash
  python main_with_pretrain.py \
      -a RandomSampling \
      -s 2000 \
      -q 10000 \
      -b 1000 \
      -d SQuAD \
      --seed 4666 \
      -t 3 \
      -g 2
```
```bash
  python main_origin.py \
      -a RandomSampling \
      -s 2000 \
      -q 10000 \
      -b 3000 \
      -d SQuAD \
      --seed 4666 \
      -t 3 \
      -g 2
```
```bash
  python main_origin.py \
      -a MarginSampling \
      -s 2000 \
      -q 10000 \
      -b 3000 \
      -d SQuAD \
      --seed 4666 \
      -t 3 \
      -g 8
```
```bash
  python main_origin.py \
      -a LeastConfidence \
      -s 2000 \
      -q 10000 \
      -b 3000 \
      -d SQuAD \
      --seed 4666 \
      -t 3 \
      -g 2
```
```bash
  python main_origin.py \
      -a EntropySampling \
      -s 2000 \
      -q 10000 \
      -b 3000 \
      -d SQuAD \
      --seed 4666 \
      -t 3 \
      -g 8
```
```bash
  python main.py \
      -a MarginSamplingDropout \
      -s 100 \
      -q 100 \
      -b 35 \
      -d SQuAD \
      --seed 4666 \
      -t 1 \
      -g 2
```
```bash
  python main.py \
      -a LeastConfidenceDropout \
      -s 100 \
      -q 100 \
      -b 35 \
      -d SQuAD \
      --seed 4666 \
      -t 1 \
      -g 2
```
```bash
  python main.py \
      -a EntropySamplingDropout \
      -s 100 \
      -q 100 \
      -b 35 \
      -d SQuAD \
      --seed 4666 \
      -t 1 \
      -g 2
```
<!-- ```bash
  python main.py \
      -a VarRatio \
      -s 100 \
      -q 100 \
      -b 35 \
      -d SQuAD \
      --seed 4666 \
      -t 1 \
      -g 2
``` -->
```bash
  python main.py \
      -a KMeansSampling \
      -s 100 \
      -q 100 \
      -b 35 \
      -d SQuAD \
      --seed 4666 \
      -t 1 \
      -g 2
```
```bash
  python main.py \
      -a KCenterGreedy \
      -s 100 \
      -q 100 \
      -b 35 \
      -d SQuAD \
      --seed 4666 \
      -t 1 \
      -g 2
```
```bash
  python main.py \
      -a BALDDropout \
      -s 100 \
      -q 100 \
      -b 35 \
      -d SQuAD \
      --seed 4666 \
      -t 1 \
      -g 2
```
```bash
  python main.py \
      -a MeanSTD \
      -s 100 \
      -q 100 \
      -b 35 \
      -d SQuAD \
      --seed 4666 \
      -t 1 \
      -g 2
```
```bash
  python main.py \
      -a BadgeSampling \
      -s 100 \
      -q 100 \
      -b 35 \
      -d SQuAD \
      --seed 4666 \
      -t 1 \
      -g 2
```
```bash
  python main.py \
      -a LossPredictionLoss \
      -s 100 \
      -q 100 \
      -b 35 \
      -d SQuAD \
      --seed 4666 \
      -t 1 \
      -g 2
```
```bash
  python main.py \
      -a CEALSampling \
      -s 100 \
      -q 100 \
      -b 35 \
      -d SQuAD \
      --seed 4666 \
      -t 1 \
      -g 2
```
