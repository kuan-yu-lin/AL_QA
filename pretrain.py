from datasets import load_dataset
from transformers import (
	default_data_collator,
	get_scheduler,
    AutoModelForQuestionAnswering
)
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

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

## load data
squad = load_dataset(DATA_NAME.lower())

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
SEED = 1127
args_input.distributed = False
if 'WORLD_SIZE' in os.environ:
    args_input.distributed = int(os.environ['WORLD_SIZE']) > 1
if args_input.distributed:
    # FOR DISTRIBUTED:  Set the device according to local_rank.
    torch.cuda.set_device(args_input.local_rank)

    # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
    # environment variables, and requires that you use init_method=`env://`.
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
# os.environ["CUDA_VISIBLE_DEVICES"] = str(3)
# torch.cuda.set_device(3)

# fix random seed
np.random.seed(SEED)
torch.manual_seed(SEED)

## device
torch.distributed.init_process_group(
    backend='nccl', init_method='env://'
)
device = torch.device("cuda", args_input.local_rank)
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

## logfile
sys.stdout = Logger(os.path.abspath('') + '/logfile/' + MODEL_NAME + '_' + DATA_NAME + '_full_dataset_normal_log.txt')
warnings.filterwarnings('ignore')

NUM_TRAIN_EPOCH = args_input.train_epochs
model_name = get_model(MODEL_NAME)

## data, network, strategy
# torch.distributed.init_process_group(backend='nccl', world_size=N, init_method='...')
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
model = DDP(model)
model = model.to(device)
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

## print info
print('Data Name:', DATA_NAME)
print('Model Name:', args_input.model)
print('Number of training data:', str(len(squad["train"])))
print('Number of validation data:', str(len(squad["validation"])))

## round 0 accuracy
to_pretrain(NUM_TRAIN_EPOCH, train_dataloader, device, model, optimizer, lr_scheduler)

acc_scores = get_pretrain_pred(eval_dataloader, device, val_features, squad['validation'])

print('testing accuracy {}'.format(acc_scores['f1']))
print('testing accuracy em {}'.format(acc_scores['exact_match']))

## save model and record acq time
timestamp = re.sub('\.[0-9]*','_',str(datetime.datetime.now())).replace(" ", "_").replace("-", "").replace(":","")
end = datetime.datetime.now()
