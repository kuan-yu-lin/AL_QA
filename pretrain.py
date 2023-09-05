from datasets import load_dataset, disable_caching
from transformers import (
	default_data_collator,
	get_scheduler,
    AutoModelForQuestionAnswering,
    AutoTokenizer
)
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch

import numpy as np

import warnings
import sys
import os
import re
import datetime

import arguments
from preprocess import *
from model import *
from utils import *

pretrain_model_dir = '/mount/arbeitsdaten31/studenten1/linku/pretrain_models' + '/' + MODEL_NAME + '_' + DATA_NAME + '_full_dataset'

CACHE_DIR = '/mount/arbeitsdaten31/studenten1/linku/.cache'

args_input = arguments.get_args()
NUM_INIT_LB = args_input.initseed
DATA_NAME = args_input.dataset_name
MODEL_BATCH = args_input.model_batch
MODEL_NAME = args_input.model
LEARNING_RATE = args_input.learning_rate
NUM_TRAIN_EPOCH = args_input.train_epochs

## load data
squad = load_dataset(DATA_NAME.lower(), cache_dir=CACHE_DIR)

## load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(get_model(MODEL_NAME))

## disable_caching
disable_caching()

## preprocess data
train_dataset = squad["train"].map(
    preprocess_training_examples,
    batched=True,
    remove_columns=squad["train"].column_names,
    fn_kwargs=dict(tokenizer=tokenizer)
)
val_dataset = squad["validation"].map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=squad["validation"].column_names,
    fn_kwargs=dict(tokenizer=tokenizer)
)
val_features = squad["validation"].map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=squad["validation"].column_names,
    fn_kwargs=dict(tokenizer=tokenizer)
)

train_dataset.set_format("torch")
val_dataset = val_dataset.remove_columns(["offset_mapping"])
val_dataset.set_format("torch")
val_features.set_format("torch")

## seed
SEED = 1127
os.environ["CUDA_VISIBLE_DEVICES"] = str(args_input.gpu)

# fix random seed
np.random.seed(SEED)
torch.manual_seed(SEED)

## device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## logfile
sys.stdout = Logger(os.path.abspath('') + '/logfile/' + MODEL_NAME + '_' + DATA_NAME + '_full_dataset_normal_log.txt')
warnings.filterwarnings('ignore')

## data, network, strategy
model = AutoModelForQuestionAnswering.from_pretrained(get_model(MODEL_NAME)).to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

start = datetime.datetime.now()

## load data to DataLoader
train_dataloader = DataLoader(
	train_dataset,
	shuffle=True,
	collate_fn=default_data_collator,
	batch_size=MODEL_BATCH,
)

eval_dataloader = DataLoader(
	val_dataset, 
	collate_fn=default_data_collator, 
	batch_size=MODEL_BATCH
)

num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = NUM_TRAIN_EPOCH * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
	"linear",
	optimizer=optimizer,
	num_warmup_steps=0,
	num_training_steps=num_training_steps,
)

scaler = amp.GradScaler()

## print info
print('Data Name:', DATA_NAME)
print('Model Name:', MODEL_NAME)
print('Number of training data:', str(len(squad["train"])))
print('Number of validation data:', str(len(squad["validation"])))

## round 0 accuracy
to_pretrain(NUM_TRAIN_EPOCH, train_dataloader, device, model, optimizer, lr_scheduler, scaler)

acc_scores = get_pred(eval_dataloader, device, val_features, squad['validation'])

print('testing accuracy {}'.format(acc_scores['f1']))
print('testing accuracy em {}'.format(acc_scores['exact_match']))

print('Time spent for training:', (datetime.datetime.now() - start))