from datasets import load_dataset
from transformers import (
    # AutoTokenizer,
    DefaultDataCollator,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer
)
import torch

import evaluate
import collections
from tqdm.auto import tqdm
import numpy as np

import warnings
import sys
import os
import re
import datetime

import arguments
from preprocess import *
from evaluations import *
from utils import *
from query import *

CACHE_DIR=os.path.abspath(os.path.expanduser('cache'))
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
NUM_QUERY = args_input.batch
NUM_INIT_LB = args_input.initseed
NUM_ROUND = int(args_input.quota / args_input.batch)
DATA_NAME = args_input.dataset_name
STRATEGY_NAME = args_input.ALstrategy

## load data
squad = load_dataset(DATA_NAME.lower())
# squad["train"] = squad["train"].shuffle(42).select(range(2000))
squad["train"] = squad["train"].select(range(3000))
squad["validation"] = squad["validation"].select(range(1000))

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

## seed
SEED = 4666
# os.environ['TORCH_HOME']='./basicmodel'
os.environ["CUDA_VISIBLE_DEVICES"] = str(args_input.gpu)

# fix random seed
np.random.seed(SEED)
torch.manual_seed(SEED)

## device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.stdout = Logger(os.path.abspath('') + '/logfile/' + DATA_NAME+ '_'  + STRATEGY_NAME + '_' + str(NUM_QUERY) + '_' + str(NUM_INIT_LB) +  '_' + str(1000) + '_normal_log.txt')
warnings.filterwarnings('ignore')

## start experiment
iteration = args_input.iteration
model_batch = args_input.model_batch

all_acc = []
acq_time = []

# repeate # iteration trials
while (iteration > 0): 
	iteration = iteration - 1

	## data, network, strategy
	net = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased").to(device)

	## set up training
	training_args = TrainingArguments(
		output_dir="./models",
		evaluation_strategy="no",
		eval_steps=100,
		logging_steps=100,
		learning_rate=1e-4,
		per_device_train_batch_size=model_batch,
		per_device_eval_batch_size=model_batch, 
		gradient_accumulation_steps=1,
		num_train_epochs=3,  # max_steps will override this value
		# max_steps=1000,  # comment out if this is not wanted
		weight_decay=0.01,
		report_to="none",
	)

	## data collator for batching
	data_collator = DefaultDataCollator()

	start = datetime.datetime.now()

	## generate initial labeled pool
	n_pool = len(train_dataset)
	labeled_idxs = np.zeros(n_pool, dtype=bool)

	tmp_idxs = np.arange(n_pool)
	np.random.shuffle(tmp_idxs)
	labeled_idxs[tmp_idxs[:args_input.initseed]] = True

	run_0_labeled_idxs = np.arange(n_pool)[labeled_idxs]

	## record acc performance 
	acc = np.zeros(NUM_ROUND + 1) # quota/batch runs + run_0

	trainer_0 = Trainer(
						model=net,
						args=training_args,
						train_dataset=train_dataset.select(indices=run_0_labeled_idxs),
						eval_dataset=val_dataset, # maybe comment out
						tokenizer=tokenizer,
						data_collator=data_collator
						)	
		
	## print info
	print(DATA_NAME)
	print(STRATEGY_NAME)
	
	## round 0 accuracy
	trainer_0.train()

	timestamp = re.sub('\.[0-9]*','_',str(datetime.datetime.now())).replace(" ", "_").replace("-", "").replace(":","")
	model_saved_dir = 'models/' + timestamp + '/' + DATA_NAME + '_' + STRATEGY_NAME + '_' + str(NUM_QUERY) + '_' + str(NUM_INIT_LB) + '_' + str(args_input.quota) + '_0'
	trainer_0.save_model(model_saved_dir)

	preds, _, _ = trainer_0.predict(val_dataset)
	start_logits, end_logits = preds
	acc[0] = compute_metrics(start_logits, end_logits, val_dataset, squad["validation"])['exact_match']

	trainer_qs = trainer_0

	print('Round 0\ntesting accuracy {}'.format(acc[0]))
	print('\n')
	
	## round 1 to rd
	for rd in range(1, NUM_ROUND+1):
		print('Round {}'.format(rd))

		## query
		if STRATEGY_NAME == 'RandomSampling':
			q_idxs = random_sampling_query(labeled_idxs)
		elif STRATEGY_NAME == 'MarginSampling':
			q_idxs = margin_sampling_query(n_pool, labeled_idxs, train_dataset, trainer_qs, squad['train'])
		elif STRATEGY_NAME == 'LeastConfidence':
			q_idxs = least_confidence_query(n_pool, labeled_idxs, train_dataset, trainer_qs, squad['train'])
		elif STRATEGY_NAME == 'EntropySampling':
			q_idxs = entropy_query()
		elif STRATEGY_NAME == 'MarginSamplingDropout':
			q_idxs = margin_sampling_dropout_query()
		elif STRATEGY_NAME == 'LeastConfidenceDropout':
			q_idxs = least_confidence_dropout_query()
		elif STRATEGY_NAME == 'EntropySamplingDropout':
			q_idxs = entropy_dropout_query()
		# elif STRATEGY_NAME == 'VarRatio':
		# 	q_idxs = var_ratio_query(n_pool, labeled_idxs, train_dataset, trainer_qs, squad['train'])
		elif STRATEGY_NAME == 'KMeansSampling':
			q_idxs = kmeans_query()
		elif STRATEGY_NAME == 'KCenterGreedy':
			q_idxs = kcenter_query()
		# elif STRATEGY_NAME == 'KCenterGreedyPCA': # not sure
		# 	q_idxs = 
		elif STRATEGY_NAME == 'BALDDropout':
			q_idxs = bayesian_query()
		elif STRATEGY_NAME == 'MeanSTD':
			q_idxs = mean_std_query()
		elif STRATEGY_NAME == 'BadgeSampling':
			q_idxs = badge_query()
		elif STRATEGY_NAME == 'LossPredictionLoss':
			# different net!
			q_idxs = loss_prediction_query()
		elif STRATEGY_NAME == 'CEALSampling':
			# why use 'CEALSampling' in STRATEGY_NAME
			q_idxs = ceal_query()
		else:
			raise NotImplementedError

		## update
		labeled_idxs[q_idxs] = True
		run_rd_labeled_idxs = np.arange(n_pool)[labeled_idxs]

		trainer_rd = Trainer(
						model=AutoModelForQuestionAnswering.from_pretrained(model_saved_dir).to(device),
						args=training_args,
						train_dataset=train_dataset.select(indices=run_rd_labeled_idxs),
						eval_dataset=val_dataset, # maybe comment out
						tokenizer=tokenizer,
						data_collator=data_collator
						)

		## train
		trainer_rd.train()
		model_saved_dir = 'models/' + timestamp + '/train_bert_squad_' + str(rd)
		trainer_rd.save_model(model_saved_dir)

		## round rd accuracy
		preds, _, _ = trainer_rd.predict(val_dataset)
		start_logits, end_logits = preds
		acc[rd] = compute_metrics(start_logits, end_logits, val_dataset, squad["validation"])['exact_match']
		print('testing accuracy {}'.format(acc[rd]))
		print('\n')

		torch.cuda.empty_cache()
	
	## print results
	print('SEED {}'.format(SEED))
	print(STRATEGY_NAME)
	print(acc)
	all_acc.append(acc)
	
	## record acq time
	end = datetime.datetime.now()
	acq_time.append(round(float((end-start).seconds),3))

# cal mean & standard deviation
acc_m = []
file_name_res_tot = DATA_NAME+ '_'  + STRATEGY_NAME + '_' + str(NUM_QUERY) + '_' + str(NUM_INIT_LB) +  '_' + str(args_input.quota) + '_normal_res_tot.txt'
file_res_tot =  open(os.path.join(os.path.abspath('') + '/results', '%s' % file_name_res_tot),'w')

file_res_tot.writelines('dataset: {}'.format(DATA_NAME) + '\n')
file_res_tot.writelines('AL strategy: {}'.format(STRATEGY_NAME) + '\n')
file_res_tot.writelines('number of labeled pool: {}'.format(NUM_INIT_LB) + '\n')
file_res_tot.writelines('number of unlabeled pool: {}'.format(len(train_dataset) - NUM_INIT_LB) + '\n')
file_res_tot.writelines('number of testing pool: {}'.format(len(val_dataset)) + '\n')
file_res_tot.writelines('batch size: {}'.format(NUM_QUERY) + '\n')
file_res_tot.writelines('quota: {}'.format(NUM_ROUND*NUM_QUERY)+ '\n')
file_res_tot.writelines('time of repeat experiments: {}'.format(args_input.iteration)+ '\n')

# result
for i in range(len(all_acc)):
	acc_m.append(get_aubc(args_input.quota, NUM_QUERY, all_acc[i]))
	print(str(i)+': '+str(acc_m[i]))
	file_res_tot.writelines(str(i)+': '+str(acc_m[i])+'\n')
mean_acc, stddev_acc = get_mean_stddev(acc_m)
mean_time, stddev_time = get_mean_stddev(acq_time)

print('mean AUBC(acc): '+str(mean_acc)+'. std dev AUBC(acc): '+str(stddev_acc))
print('mean time: '+str(mean_time)+'. std dev time: '+str(stddev_time))

file_res_tot.writelines('mean acc: '+str(mean_acc)+'. std dev acc: '+str(stddev_acc)+'\n')
file_res_tot.writelines('mean time: '+str(mean_time)+'. std dev acc: '+str(stddev_time)+'\n')

# save result
file_name_res = DATA_NAME+ '_'  + STRATEGY_NAME + '_' + str(NUM_QUERY) + '_' + str(NUM_INIT_LB) +  '_' + str(args_input.quota) + '_normal_res.txt'
file_res =  open(os.path.join(os.path.abspath('') + '/results', '%s' % file_name_res),'w')


file_res.writelines('dataset: {}'.format(DATA_NAME) + '\n')
file_res.writelines('AL strategy: {}'.format(STRATEGY_NAME) + '\n')
file_res.writelines('number of labeled pool: {}'.format(NUM_INIT_LB) + '\n')
# file_res.writelines('number of unlabeled pool: {}'.format(dataset.n_pool - NUM_INIT_LB) + '\n')
file_res.writelines('number of unlabeled pool: {}'.format(len(train_dataset) - NUM_INIT_LB) + '\n')
# file_res.writelines('number of testing pool: {}'.format(dataset.n_test) + '\n')
file_res.writelines('number of testing pool: {}'.format(len(val_dataset)) + '\n')
file_res.writelines('batch size: {}'.format(NUM_QUERY) + '\n')
file_res.writelines('quota: {}'.format(NUM_ROUND*NUM_QUERY)+ '\n')
file_res.writelines('time of repeat experiments: {}'.format(args_input.iteration)+ '\n')
avg_acc = np.mean(np.array(all_acc),axis=0)
for i in range(len(avg_acc)):
	tmp = 'Size of training set is ' + str(NUM_INIT_LB + i*NUM_QUERY) + ', ' + 'accuracy is ' + str(round(avg_acc[i],4)) + '.' + '\n'
	file_res.writelines(tmp)

file_res.close()
file_res_tot.close()
