from datasets import load_dataset
from transformers import (
	default_data_collator,
	get_scheduler,
    AutoModelForQuestionAnswering
)
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch

import numpy as np

# import warnings
# import sys
import os
import re
import datetime

import arguments
from preprocess import *
from model import *
from utils import *
# from query import *

model_dir = '/mount/arbeitsdaten31/studenten1/linku/models'

CACHE_DIR = '/mount/arbeitsdaten31/studenten1/linku/cache'
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_MODULES_CACHE'] = CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = CACHE_DIR

# args_input_ALstrategy = 'RandomSampling'
# args_input_initseed = 100 # 1000
# args_input_quota = 100 # 1000
# args_input_batch = 35 # 128
# args_input_dataset_name = 'SQuAD'
# args_input_iteration = 3

args_input = arguments.get_args()
# NUM_QUERY = args_input.batch
NUM_INIT_LB = args_input.initseed
# NUM_ROUND = int(args_input.quota / args_input.batch)
DATA_NAME = args_input.dataset_name
# STRATEGY_NAME = args_input.ALstrategy

## load data
squad = load_dataset(DATA_NAME.lower())
# squad["train"] = squad["train"].shuffle(42).select(range(2000))
# squad["train"] = squad["train"] # for init=4000
# squad["validation"] = squad["validation"] for init=4000
squad["train"] = squad["train"].select(range(4000))
squad["validation"] = squad["validation"].select(range(1500))

## preprocess data
train_dataset = squad["train"].map(
    preprocess_training_examples,
    batched=True,
    remove_columns=squad["train"].column_names,
)
val_dataset = squad["validation"].map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=squad["validation"].column_names,
)
val_features = squad["validation"].map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=squad["validation"].column_names,
)

train_dataset.set_format("torch")
val_dataset = val_dataset.remove_columns(["offset_mapping"])
val_dataset.set_format("torch")
val_features.set_format("torch")

## seed
SEED = 4666
os.environ["CUDA_VISIBLE_DEVICES"] = str(args_input.gpu)

# fix random seed
np.random.seed(SEED)
torch.manual_seed(SEED)

## device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_TRAIN_EPOCH = args_input.train_epochs
model_name = get_model(args_input.model)

## data, network, strategy
model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
optimizer = AdamW(model.parameters(), lr=1e-4)

start = datetime.datetime.now()

## generate initial labeled pool
n_pool = len(train_dataset)
labeled_idxs = np.zeros(n_pool, dtype=bool)

tmp_idxs = np.arange(n_pool)
np.random.shuffle(tmp_idxs)
labeled_idxs[tmp_idxs[:NUM_INIT_LB]] = True

run_0_labeled_idxs = np.arange(n_pool)[labeled_idxs]

## load the selected train data to DataLoader
train_dataloader = DataLoader(
	train_dataset.select(indices=run_0_labeled_idxs),
	shuffle=True,
	collate_fn=default_data_collator,
	batch_size=8,
)

eval_dataloader = DataLoader(
	val_dataset, 
	collate_fn=default_data_collator, 
	batch_size=8
)

num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = NUM_TRAIN_EPOCH * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
	"linear",
	optimizer=optimizer,
	num_warmup_steps=0,
	num_training_steps=num_training_steps,
)

## print info
print(DATA_NAME)

## round 0 accuracy
to_train(NUM_TRAIN_EPOCH, train_dataloader, device, model, optimizer, lr_scheduler, record_loss=True)

acc = get_pred(eval_dataloader, device, val_features, squad['validation'], record_loss=True)['f1']
print('\nTest set: F1 score: {:.4f}\n'.format(acc))

## save model and record acq time
timestamp = re.sub('\.[0-9]*','_',str(datetime.datetime.now())).replace(" ", "_").replace("-", "").replace(":","")
pretrain_model_dir = model_dir + '/' + DATA_NAME + '_' + str(NUM_INIT_LB) + '_' + args_input.model
os.makedirs(pretrain_model_dir, exist_ok=True)
end = datetime.datetime.now()

final_model = AutoModelForQuestionAnswering.from_pretrained(model_dir).to(device)
model_to_save = final_model.module if hasattr(final_model, 'module') else final_model 
model_to_save.save_pretrained(pretrain_model_dir)
