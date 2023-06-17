# ActiveLearning_QuestionAnswering

## run
```
  python Random.py \
      -a RandomSampling \
      -s 100 \
      -q 100 \
      -b 35 \
      -d SQuAD \
      --seed 4666 \
      -t 1 \
      -g 2
```

```
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