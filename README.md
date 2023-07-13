# ActiveLearning_QuestionAnswering

## To-do list:
- [x] set up all with main_with_pretrain
- [x] ~~consider to saperate the first random training (5 mins difference with init=2000 batch=8)~~
- [ ] implement with roberta-base
- [ ] how long does it takes with bert-large and roberta-large
- [ ] implement the option for different datasets

## cache setup

https://huggingface.co/docs/datasets/v1.12.0/cache.html

## folder needed for development

- logfile/
- /mount/arbeitsdaten31/studenten1/linku/models
- results/

## select dataset
Model: Bert-base Bert-large(how long) Roberta-large Roberta-base(http://arxiv.org/abs/2101.00438)

### source domain

source domain: SQuAD v1.1
Train: 87599	
Test: 10570

source domain: Natural Question
Train: 307373
Test: 7830

### target domain

target domain: BioASQ
Train: 3.27k
Test: 4.95k

target domain: NewsQA
link: https://huggingface.co/datasets/newsqa
Train: 92549
Val: 5166
Test: 5126

target domain: SearchQA
  We start not from an existing article and generate a question-answer pair, but start from an existing question-answer pair, crawled from J! Archive, and augment it with text snippets retrieved by Google.
link: https://huggingface.co/datasets/search_qa
Train: 151295	
Val: 21613
Test: 43228

target domain: DROP
  A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs
link: https://huggingface.co/datasets/drop
Train: 77409
Test: 9536

target domain: DuoRC
  The DuoRC dataset is an English language dataset of questions and answers gathered from crowdsourced AMT workers on Wikipedia and IMDb movie plots.
link: https://huggingface.co/datasets/duorc
Train: 69.5k
Val: 15.6k
Test: 15.9k
