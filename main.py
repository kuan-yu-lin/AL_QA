from datasets import load_dataset, disable_caching
from transformers import (
	default_data_collator,
	get_scheduler,
    AutoModelForQuestionAnswering,
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
from model import to_train, get_pred
from utils import *
from query import *

args_input = arguments.get_args()
NUM_QUERY = args_input.batch
NUM_INIT_LB = args_input.initseed
ITERATION = int(args_input.quota / args_input.batch)
DATA_NAME = args_input.dataset_name
STRATEGY_NAME = args_input.ALstrategy
MODEL_NAME = args_input.model
LEARNING_RATE = args_input.learning_rate
EXPE_ROUND = args_input.expe_round
MODEL_BATCH = args_input.model_batch
NUM_TRAIN_EPOCH = args_input.train_epochs
LOW_RES = args_input.low_resource
UNIQ_CONTEXT = args_input.unique_context

model_dir = '/mount/arbeitsdaten31/studenten1/linku/models'
CACHE_DIR = '/mount/arbeitsdaten31/studenten1/linku/.cache'

if LOW_RES:
	## set dir
	pretrain_model_dir = '/mount/arbeitsdaten31/studenten1/linku/pretrain_models' + '/' + MODEL_NAME + '_SQuAD_full_dataset_lr_3e-5'
	strategy_model_dir = model_dir + '/lowRes_' + str(args_input.quota) + '_' + STRATEGY_NAME + '_' + MODEL_NAME +  '_' + DATA_NAME
	
	## load data
	train_data, val_data = load_dataset_mrqa(DATA_NAME.lower())

else:
	## set dir
	strategy_model_dir = model_dir + '/' + str(NUM_INIT_LB) + '_' + str(args_input.quota) + '_' + STRATEGY_NAME + '_' + MODEL_NAME +  '_' + DATA_NAME
	## load data
	squad = load_dataset(DATA_NAME.lower(), cache_dir=CACHE_DIR)
	if args_input.toy_exp:
		print('Use 4000 training data and 1500 testing data.')
		train_data = squad["train"].select(range(4000))
		val_data = squad["validation"]
	else:
		train_data = squad["train"]
		val_data = squad["validation"]
		print('Use full training data and full testing data.')

## disable_caching
disable_caching()

## preprocess data
train_dataset, train_features, val_dataset, val_features = preprocess_data(train_data, val_data)
context_dict = get_context_id(train_data)
print('len(context_dict):', len(context_dict)) # len(context_dict): 18891

# get the number of extra data after preprocessing
extra = len(train_dataset) - len(train_data)

## seed
SEED = 1127
os.environ["CUDA_VISIBLE_DEVICES"] = str(args_input.gpu)

# fix random seed
np.random.seed(SEED)
torch.manual_seed(SEED)

## device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if LOW_RES:
	sys.stdout = Logger(os.path.abspath('') + '/logfile_lowRes/' + str(args_input.quota) + '_' + STRATEGY_NAME + '_' + MODEL_NAME + '_' + DATA_NAME + '_normal_log.txt')
else:
	sys.stdout = Logger(os.path.abspath('') + '/logfile/' + str(NUM_INIT_LB) + '_' + str(args_input.quota) + '_' + STRATEGY_NAME + '_' + MODEL_NAME + '_' + DATA_NAME + '_normal_log.txt')
warnings.filterwarnings('ignore')

## start experiment
all_acc = []
acq_time = []

begin = datetime.datetime.now()

## repeate experiment trials
while (EXPE_ROUND > 0): 
	EXPE_ROUND = EXPE_ROUND - 1
	
	start = datetime.datetime.now()

	## record acc performance 
	acc = np.zeros(ITERATION + 1) # quota/batch runs + iter_0
	acc_em = np.zeros(ITERATION + 1)

	## generate initial labeled pool
	n_pool = len(train_dataset)
	print('n_pool:', n_pool)
	labeled_idxs = np.zeros(n_pool, dtype=bool)

	if LOW_RES:
		print('in low res setting')
		save_model(device, pretrain_model_dir, strategy_model_dir)
		## print info
		print(DATA_NAME)
		print(STRATEGY_NAME)
	else:
		print('not in low res setting')
		tmp_idxs = np.arange(n_pool)
		np.random.shuffle(tmp_idxs)
		
		if UNIQ_CONTEXT:
			print('in uc setting')
			tmp_idxs = tmp_idxs[:NUM_INIT_LB+extra]
			print('number of idxs:', len(tmp_idxs))
			uc_tmp_idxs = get_unique_context(tmp_idxs, train_features, context_dict) # len() = almost num_query + extra
			print('number of uc idxs:', len(uc_tmp_idxs))
			iter_0_labeled_idxs = get_unique_sample(labeled_idxs, uc_tmp_idxs, n_pool, train_features)
			c_id = get_final_c_id(iter_0_labeled_idxs, train_features, context_dict) # len() = num_query
		else:
			print('not in uc setting')
			iter_0_labeled_idxs = get_unique_sample(labeled_idxs, tmp_idxs, n_pool, train_features)

		## load the selected train data to DataLoader
		train_dataloader = DataLoader(
			train_dataset.select(indices=iter_0_labeled_idxs),
			shuffle=True,
			collate_fn=default_data_collator,
			batch_size=MODEL_BATCH,
		)

		num_update_steps_per_epoch = len(train_dataloader)
		num_training_steps = NUM_TRAIN_EPOCH * num_update_steps_per_epoch

		## network
		model = AutoModelForQuestionAnswering.from_pretrained(get_model(MODEL_NAME)).to(device)
		optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
		
		lr_scheduler = get_scheduler(
			"linear",
			optimizer=optimizer,
			num_warmup_steps=0,
			num_training_steps=num_training_steps,
		)
		## print info
		print(DATA_NAME)
		print(STRATEGY_NAME)

		## iteration 0 accuracy
		to_train(NUM_TRAIN_EPOCH, train_dataloader, device, model, optimizer, lr_scheduler)

	## load the selected validation data to DataLoader
	eval_dataloader = DataLoader(
		val_dataset, 
		collate_fn=default_data_collator, 
		batch_size=MODEL_BATCH
	)

	acc_scores_0 = get_pred(eval_dataloader, device, val_features, val_data) # add i=1 to use model from models_dir
	acc[0] = acc_scores_0['f1']
	acc_em[0] = acc_scores_0['exact_match']

	print('Round 0\ntesting accuracy {}'.format(acc[0]))
	print('testing accuracy em {}'.format(acc_em[0]))
	time = datetime.datetime.now()
	print('Time spent for init training:', (time - start))
	print('\n')
	
	## iteration 1 to i
	for i in range(1, ITERATION+1):
		print('Iteraion {} in experiment round {}'.format(i, 5 - EXPE_ROUND))

		## use total_query (NUM_QUERY + extra) to query instead of just NUM_QUERY
		total_query = NUM_QUERY + extra
		
		## query
		if STRATEGY_NAME == 'RandomSampling':
			q_idxs = random_sampling_query(labeled_idxs, total_query)
		elif STRATEGY_NAME == 'MarginSampling':
			q_idxs = margin_sampling_query(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, total_query)
		elif STRATEGY_NAME == 'LeastConfidence':
			q_idxs = least_confidence_query(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, total_query)
		elif STRATEGY_NAME == 'EntropySampling':
			q_idxs = entropy_query(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, total_query)
		elif STRATEGY_NAME == 'MarginSamplingDropout':
			q_idxs = margin_sampling_dropout_query(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, total_query)
		elif STRATEGY_NAME == 'LeastConfidenceDropout':
			q_idxs = least_confidence_dropout_query(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, total_query)
		elif STRATEGY_NAME == 'EntropySamplingDropout':
			q_idxs = entropy_dropout_query(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, total_query)
		elif STRATEGY_NAME == 'VarRatio':
			q_idxs = var_ratio_query(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, total_query)
		elif STRATEGY_NAME == 'BALDDropout':
			q_idxs = bald_query(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, total_query)
		elif STRATEGY_NAME == 'MeanSTD':
			q_idxs = mean_std_query(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, total_query)
		elif STRATEGY_NAME == 'KMeansSampling':
			q_idxs = kmeans_query(n_pool, labeled_idxs, train_dataset, device, total_query)
		elif STRATEGY_NAME == 'KCenterGreedy':
			q_idxs = kcenter_greedy_query(n_pool, labeled_idxs, train_dataset, device, total_query)
		elif STRATEGY_NAME == 'BadgeSampling':
			q_idxs = badge_query(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, total_query)
		else:
			raise NotImplementedError

		print('Time spent for querying:', (datetime.datetime.now() - time))
		time = datetime.datetime.now()

		if UNIQ_CONTEXT:
			print('in uc setting')
			if LOW_RES:
				print('in lr setting')
				print('num of idxs:', len(q_idxs))
				uc_q_idxs = get_unique_context(q_idxs, train_features, context_dict)
				print('num of uc idxs:', len(uc_q_idxs))
			else:
				print('not in lr setting')
				print('num of idxs:', len(q_idxs))
				uc_q_idxs = get_unique_context(q_idxs, train_features, context_dict, c_id)
				print('num of ucidxs:', len(uc_q_idxs))
			
			iter_i_labeled_idxs = get_unique_sample(labeled_idxs, uc_q_idxs, n_pool, train_features, i)
			c_id = get_final_c_id(iter_i_labeled_idxs, train_features, context_dict)
		else:
			print('not in uc setting')
			iter_i_labeled_idxs = get_unique_sample(labeled_idxs, q_idxs, n_pool, train_features, i)
		 
		train_dataloader_i = DataLoader(
			train_dataset.select(indices=iter_i_labeled_idxs),
			shuffle=True,
			collate_fn=default_data_collator,
			batch_size=MODEL_BATCH,
		)

		num_update_steps_per_epoch_i = len(train_dataloader_i)
		num_training_steps_i = NUM_TRAIN_EPOCH * num_update_steps_per_epoch_i

		model_i = AutoModelForQuestionAnswering.from_pretrained(strategy_model_dir).to(device)
		optimizer_i = AdamW(model_i.parameters(), lr=LEARNING_RATE)
		
		lr_scheduler_i = get_scheduler(
			"linear",
			optimizer=optimizer_i,
			num_warmup_steps=0,
			num_training_steps=num_training_steps_i,
		)
		
		## train
		to_train(NUM_TRAIN_EPOCH, train_dataloader_i, device, model_i, optimizer_i, lr_scheduler_i)

		## iteration i accuracy
		print('iter_{} get_pred!'.format(i))
		acc_scores_i = get_pred(eval_dataloader, device, val_features, val_data)
		acc[i] = acc_scores_i['f1']
		acc_em[i] = acc_scores_i['exact_match']
		print('testing accuracy {}'.format(acc[i]))
		print('testing accuracy em {}'.format(acc_em[i]))
		print('Time spent for training after querying:', (datetime.datetime.now() - time))
		time = datetime.datetime.now()
		print('\n')

		torch.cuda.empty_cache()
	
	## print results
	print('SEED {}'.format(SEED))
	print(STRATEGY_NAME)
	print(acc)
	all_acc.append(acc)
	
	## record acq time
	timestamp = re.sub('\.[0-9]*', '_', str(datetime.datetime.now())).replace(" ", "_").replace("-", "").replace(":","")
	end = datetime.datetime.now()
	acq_time.append(round(float((end-start).seconds), 3))

# cal mean & standard deviation
total_time = datetime.datetime.now() - begin
print('Time spent in total:', total_time)
acc_m = []
if LOW_RES:
	file_name_res = str(args_input.quota) + '_' + STRATEGY_NAME + '_new_' + MODEL_NAME + '_' + DATA_NAME + '_res.txt'
	file_res =  open(os.path.join(os.path.abspath('') + '/results_lowRes', '%s' % file_name_res),'w')
else:
	file_name_res = str(NUM_INIT_LB) + '_' + str(args_input.quota) + '_' + STRATEGY_NAME + '_' + MODEL_NAME + '_' + DATA_NAME + '_normal_res.txt'
	file_res =  open(os.path.join(os.path.abspath('') + '/results', '%s' % file_name_res),'w')

file_res.writelines('dataset: {}'.format(DATA_NAME) + '\n')
file_res.writelines('AL strategy: {}'.format(STRATEGY_NAME) + '\n')
if not LOW_RES: file_res.writelines('number of labeled pool: {}'.format(NUM_INIT_LB) + '\n')
file_res.writelines('number of unlabeled pool: {}'.format(len(train_dataset) - NUM_INIT_LB) + '\n')
file_res.writelines('number of testing pool: {}'.format(len(val_dataset)) + '\n')
file_res.writelines('batch size: {}'.format(NUM_QUERY) + '\n')
file_res.writelines('quota: {}'.format(ITERATION * NUM_QUERY) + '\n')
file_res.writelines('learning rate: {}'.format(LEARNING_RATE) + '\n')
file_res.writelines('training batch size: {}'.format(MODEL_BATCH) + '\n')
file_res.writelines('time of repeat experiments: {}'.format(args_input.expe_round) + '\n')
file_res.writelines('The experiment started at {}'.format(begin) + '\n')
file_res.writelines('The experiment ended at {}'.format(end) + '\n')
file_res.writelines('Time spent in total: {}'.format(total_time) + '\n')

# save result
file_res.writelines('\nAUBC in each experiment round.\n')
for i in range(len(all_acc)):
	acc_m.append(get_aubc(args_input.quota, NUM_QUERY, all_acc[i]))
	print('The experiment round ' + str(i) + ': ' + str(acc_m[i]))
	file_res.writelines(str(i) + ': ' + str(acc_m[i]) + '\n')
mean_acc, stddev_acc = get_mean_stddev(acc_m)
mean_time, stddev_time = get_mean_stddev(acq_time)

print('mean AUBC(acc): ' + str(mean_acc) + '. std dev AUBC(acc): ' + str(stddev_acc))
print('mean time: ' + str(mean_time) + '. std dev time: ' + str(stddev_time) + '\n')

avg_acc = np.mean(np.array(all_acc), axis=0)
stddev_i_acc = np.std(np.array(all_acc), axis=0)
for i in range(len(avg_acc)):
	if LOW_RES:
		tmp = 'When the size of training set is ' + str(i * NUM_QUERY + NUM_QUERY) + ', ' + 'average accuracy is ' + str(round(avg_acc[i], 4)) + ', ' + 'std dev is ' + str(round(stddev_i_acc[i], 4)) + '.' + '\n'
	else:
		tmp = 'When the size of training set is ' + str(NUM_INIT_LB + i * NUM_QUERY) + ', ' + 'average accuracy is ' + str(round(avg_acc[i], 4)) + ', ' + 'std dev is ' + str(round(stddev_i_acc[i], 4)) + '.' + '\n'
	file_res.writelines(tmp)

file_res.writelines('mean acc: ' + str(mean_acc) + '. std dev acc: ' + str(stddev_acc) + '\n')
file_res.writelines('mean time: ' + str(datetime.timedelta(seconds=mean_time)) + '. std dev acc: ' + str(datetime.timedelta(seconds=stddev_time)) + '\n')

file_res.close()