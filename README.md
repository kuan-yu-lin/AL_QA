# ActiveLearning_QuestionAnswering

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
  python main.py \
      -a RandomSampling \
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
      -a MarginSampling \
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
      -a LeastConfidence \
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
      -a EntropySampling \
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
