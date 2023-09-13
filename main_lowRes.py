from datasets import disable_caching
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
from model_lowRes import to_train_lowRes, get_pred_lowRes
from utils import *
from query import *

args_input = arguments.get_args()
NUM_QUERY = args_input.batch
# NUM_INIT_LB = args_input.initseed
ITERATION = int(args_input.quota / args_input.batch)
DATA_NAME = args_input.dataset_name
STRATEGY_NAME = args_input.ALstrategy
MODEL_NAME = args_input.model
LEARNING_RATE = args_input.learning_rate
EXPE_ROUND = args_input.expe_round
MODEL_BATCH = args_input.model_batch
NUM_TRAIN_EPOCH = args_input.train_epochs

model_dir = '/mount/arbeitsdaten31/studenten1/linku/models'
pretrain_model_dir = '/mount/arbeitsdaten31/studenten1/linku/pretrain_models' + '/' + MODEL_NAME + '_SQuAD_full_dataset'
strategy_model_dir = model_dir + '/lowRes_' + str(args_input.quota) + '_' + STRATEGY_NAME + '_' + MODEL_NAME +  '_' + DATA_NAME

CACHE_DIR = '/mount/arbeitsdaten31/studenten1/linku/.cache'

## load data
train_data, val_data = load_dataset_mrqa(DATA_NAME.lower())
# cache_dir=CACHE_DIR is build-in in the func.

## load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(get_model(MODEL_NAME))

## disable_caching
disable_caching()

## preprocess data
# TODO: change to 
train_dataset = train_data.map(
    preprocess_training_examples_lowRes,
    batched=True,
    remove_columns=train_data.column_names,
	fn_kwargs=dict(tokenizer=tokenizer)
)
train_features = train_data.map(
    preprocess_training_features_lowRes,
    batched=True,
    remove_columns=train_data.column_names,
	fn_kwargs=dict(tokenizer=tokenizer)
)
val_dataset = val_data.map(
    preprocess_validation_examples_lowRes,
    batched=True,
    remove_columns=val_data.column_names,
	fn_kwargs=dict(tokenizer=tokenizer)
)
val_features = val_data.map(
    preprocess_validation_examples_lowRes,
    batched=True,
    remove_columns=val_data.column_names,
	fn_kwargs=dict(tokenizer=tokenizer)
)

train_dataset.set_format("torch")
train_features.set_format("torch")
val_dataset = val_dataset.remove_columns(["offset_mapping"])
val_dataset.set_format("torch")
val_features.set_format("torch")

# get the number of extra data after preprocessing
extra = min(NUM_QUERY, len(train_dataset) - len(train_data))

## seed
SEED = 1127
os.environ["CUDA_VISIBLE_DEVICES"] = str(args_input.gpu)

# fix random seed
np.random.seed(SEED)
torch.manual_seed(SEED)

## device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.stdout = Logger(os.path.abspath('') + '/logfile_lowRes/' + str(args_input.quota) + '_' + STRATEGY_NAME + '_' + MODEL_NAME + '_' + DATA_NAME + '_normal_log.txt')
warnings.filterwarnings('ignore')

## start experiment
all_acc = []
acq_time = []

begin = datetime.datetime.now()

## repeate experiment trials
while (EXPE_ROUND > 0): 
	EXPE_ROUND = EXPE_ROUND - 1

	start = datetime.datetime.now()
	
	## generate initial labeled pool
	n_pool = len(train_dataset)
	labeled_idxs = np.zeros(n_pool, dtype=bool)
	
	## record acc performance 
	acc = np.zeros(ITERATION) # quota/batch runs
	acc_em = np.zeros(ITERATION)

	## load the selected train data to DataLoader
	eval_dataloader = DataLoader(
		val_dataset, 
		collate_fn=default_data_collator, 
		batch_size=MODEL_BATCH
	)

	time = datetime.datetime.now()
	
	## iteration 1 to i
	for i in range(1, ITERATION+1):
		print('Iteraion {} in experiment round {}'.format(i, 5 - EXPE_ROUND))

		## use total_query (NUM_QUERY + extra) to query instead of just NUM_QUERY
		total_query = NUM_QUERY + extra

		## query
		if STRATEGY_NAME == 'RandomSampling':
			q_idxs = random_sampling_query(labeled_idxs, total_query)
		elif STRATEGY_NAME == 'MarginSampling':
			q_idxs = margin_sampling_query(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, total_query, i)
		elif STRATEGY_NAME == 'LeastConfidence':
			q_idxs = least_confidence_query(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, total_query, i)
		elif STRATEGY_NAME == 'EntropySampling':
			q_idxs = entropy_query(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, total_query, i)
		elif STRATEGY_NAME == 'MarginSamplingDropout':
			q_idxs = margin_sampling_dropout_query(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, total_query, i)
		elif STRATEGY_NAME == 'LeastConfidenceDropout':
			q_idxs = least_confidence_dropout_query(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, total_query, i)
		elif STRATEGY_NAME == 'EntropySamplingDropout':
			q_idxs = entropy_dropout_query(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, total_query, i)
		elif STRATEGY_NAME == 'VarRatio':
			q_idxs = var_ratio_query(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, total_query, i)
		elif STRATEGY_NAME == 'BALDDropout':
			q_idxs = bald_query(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, total_query, i)
		elif STRATEGY_NAME == 'MeanSTD':
			q_idxs = mean_std_query(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, total_query, i)
		elif STRATEGY_NAME == 'KMeansSampling':
			q_idxs = kmeans_query(n_pool, labeled_idxs, train_dataset, device, total_query, i)
		elif STRATEGY_NAME == 'KCenterGreedy':
			q_idxs = kcenter_greedy_query(n_pool, labeled_idxs, train_dataset, device, total_query, i)
		# elif STRATEGY_NAME == 'KCenterGreedyPCA': # not sure
		# 	q_idxs = kcenter_greedy_PCA_query(n_pool, labeled_idxs, train_dataset, device, total_query, i)
		elif STRATEGY_NAME == 'BadgeSampling':
			q_idxs = badge_query(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, total_query, i)
		# elif STRATEGY_NAME == 'LossPredictionLoss':
		# 	# different net!
		# 	q_idxs = loss_prediction_query()
		# elif STRATEGY_NAME == 'CEALSampling':
		# 	# why use 'CEALSampling' in STRATEGY_NAME
		# 	q_idxs = ceal_query()
		else:
			raise NotImplementedError

		print('Time spent for querying:', (datetime.datetime.now() - time))
		time = datetime.datetime.now()

		## update
    
		## goal of total query data: NUM_QUERY * i
		num_set_query_i = NUM_QUERY * i

		# difference_i = 0
		# num_set_ex_id_i = 0
		labeled_idxs[q_idxs[:NUM_QUERY]] = True
		run_i_labeled_idxs = np.arange(n_pool)[labeled_idxs]

		run_i_samples = train_features.select(indices=run_i_labeled_idxs)
		num_set_ex_id_i = len(set(run_i_samples['example_id']))

		difference_i = num_set_query_i - num_set_ex_id_i

		if difference_i:
			labeled_idxs[q_idxs[NUM_QUERY:(NUM_QUERY + difference_i)]] = True
			run_i_labeled_idxs = np.arange(n_pool)[labeled_idxs]

		train_dataloader_i = DataLoader(
			train_dataset.select(indices=run_i_labeled_idxs),
			shuffle=True,
			collate_fn=default_data_collator,
			batch_size=MODEL_BATCH,
		)

		num_update_steps_per_epoch_i = len(train_dataloader_i)
		num_training_steps_i = NUM_TRAIN_EPOCH * num_update_steps_per_epoch_i

		if i == 1:
			print('Use pretrain model in iteration ', i)
			model_i = AutoModelForQuestionAnswering.from_pretrained(pretrain_model_dir).to(device)
		else:
			print('Use strategy model in iteration ', i)
			model_i = AutoModelForQuestionAnswering.from_pretrained(strategy_model_dir).to(device)

		optimizer_i = AdamW(model_i.parameters(), lr=LEARNING_RATE)
		
		lr_scheduler_i = get_scheduler(
			"linear",
			optimizer=optimizer_i,
			num_warmup_steps=0,
			num_training_steps=num_training_steps_i,
		)
		
		## train
		to_train_lowRes(NUM_TRAIN_EPOCH, train_dataloader_i, device, model_i, optimizer_i, lr_scheduler_i)

		## iteration i accuracy
		print('iter_{} get_pred!'.format(i))
		acc_scores_i = get_pred_lowRes(eval_dataloader, device, val_features, val_data)
		acc[i-1] = acc_scores_i['f1']
		acc_em[i-1] = acc_scores_i['exact_match']
		print('testing accuracy {}'.format(acc[i-1]))
		print('testing accuracy em {}'.format(acc_em[i-1]))
		print('Time spent for training after querying:', (datetime.datetime.now() - time))
		time = datetime.datetime.now()
		print('\n')

		torch.cuda.empty_cache()
	
	## print results
	print('SEED {}'.format(SEED))
	print(STRATEGY_NAME)
	print(acc)
	all_acc.append(acc)
	
	## save model and record acq time
	timestamp = re.sub('\.[0-9]*','_',str(datetime.datetime.now())).replace(" ", "_").replace("-", "").replace(":","")
	final_model_dir = model_dir + '/' + timestamp + DATA_NAME+ '_'  + STRATEGY_NAME + '_' + str(NUM_QUERY) + '_' + str(args_input.quota)
	os.makedirs(final_model_dir, exist_ok=True)
	end = datetime.datetime.now()
	acq_time.append(round(float((end-start).seconds), 3))

	final_model = AutoModelForQuestionAnswering.from_pretrained(strategy_model_dir).to(device)
	model_to_save = final_model.module if hasattr(final_model, 'module') else final_model 
	model_to_save.save_pretrained(final_model_dir)

# cal mean & standard deviation
print('Time spent in total:', (datetime.datetime.now() - begin))
acc_m = []
file_name_res = str(args_input.quota) + '_' + STRATEGY_NAME + '_' + MODEL_NAME + '_' + DATA_NAME + '_normal_res.txt'
file_res =  open(os.path.join(os.path.abspath('') + '/results', '%s' % file_name_res),'w')

file_res.writelines('dataset: {}'.format(DATA_NAME) + '\n')
file_res.writelines('AL strategy: {}'.format(STRATEGY_NAME) + '\n')
file_res.writelines('number of unlabeled pool: {}'.format(len(train_dataset)) + '\n')
file_res.writelines('number of testing pool: {}'.format(len(val_dataset)) + '\n')
file_res.writelines('batch size: {}'.format(NUM_QUERY) + '\n')
file_res.writelines('quota: {}'.format(ITERATION * NUM_QUERY) + '\n')
file_res.writelines('learning rate: {}'.format(LEARNING_RATE) + '\n')
file_res.writelines('training batch size: {}'.format(MODEL_BATCH) + '\n')
file_res.writelines('time of repeat experiments: {}'.format(args_input.expe_round) + '\n')

# save result
file_res.writelines('\nAUBC in each experiment round.')
for i in range(len(all_acc)):
	acc_m.append(get_aubc(args_input.quota, NUM_QUERY, all_acc[i]))
	print(str(i) + ': ' + str(acc_m[i]))
	file_res.writelines(str(i) + ': ' + str(acc_m[i]) + '\n')
mean_acc, stddev_acc = get_mean_stddev(acc_m)
mean_time, stddev_time = get_mean_stddev(acq_time)

print('mean AUBC(acc): ' + str(mean_acc) + '. std dev AUBC(acc): ' + str(stddev_acc))
print('mean time: ' + str(mean_time) + '. std dev time: ' + str(stddev_time))

avg_acc = np.mean(np.array(all_acc),axis=0)
for i in range(len(avg_acc)):
	tmp = 'Size of training set is ' + str(i * NUM_QUERY) + ', ' + 'accuracy is ' + str(round(avg_acc[i],4)) + '.' + '\n'
	file_res.writelines(tmp)

file_res.writelines('mean acc: ' + str(mean_acc) + '. std dev acc: ' + str(stddev_acc) + '\n')
file_res.writelines('mean time: ' + str(mean_time) + '. std dev acc: ' + str(stddev_time) + '\n')

file_res.close()
