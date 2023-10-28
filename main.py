from datasets import load_dataset, disable_caching
from transformers import default_data_collator, get_scheduler, AutoModelForQuestionAnswering
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
import numpy as np
import json
import warnings
import sys
import os
import re
import datetime

import arguments
from model import to_train, get_pred
from utils import *
from strategies.randomSampling import random_sampling
from strategies.margin import margin
from strategies.lc import least_confidence
from strategies.entropy import entropy
from strategies.marginDropout import margin_dropout
from strategies.lcDropout import least_confidence_dropout
from strategies.entropyDropout import entropy_dropout
from strategies.kcenter import kcenter
from strategies.kmeans import kmeans
from strategies.meanSTD import mean_std
from strategies.bald import bald
from strategies.badge import badge
from strategies.batchBald import batch_bald

args_input = arguments.get_args()
NUM_QUERY = args_input.batch
NUM_INIT_LB = args_input.initseed
ITERATION = int(args_input.quota / args_input.batch)
DATA_NAME = args_input.dataset
STRATEGY_NAME = args_input.ALstrategy
MODEL_NAME = args_input.model
LEARNING_RATE = args_input.learning_rate
EXPE_ROUND = args_input.expe_round
MODEL_BATCH = args_input.model_batch
NUM_TRAIN_EPOCH = args_input.train_epochs
## exp setting
EXP_ID = str(args_input.exp_id)
LOW_RES = args_input.low_resource
UNIQ_CONTEXT = args_input.unique_context

## set dir
if args_input.dev_mode:
	MODEL_DIR = os.path.abspath('') + '/dev_models'
else:
	MODEL_DIR = os.path.abspath('') + '/models'
CACHE_DIR = os.path.abspath('') + '/.cache'
file_name_res = 'exp' + EXP_ID + '.json'
OUTPUT_DIR = os.path.join(os.path.abspath('') + '/results', '%s' % file_name_res)

if LOW_RES:
	init_pool = 0
	setting = 'low resource'
	## set dir
	pretrain_model_dir = os.path.abspath('') + '/pretrain_models' + '/' + MODEL_NAME + '_SQuAD_full_dataset_lr_3e-5'
	strategy_model_dir = MODEL_DIR + '/lowRes_' + str(args_input.quota) + '_' + STRATEGY_NAME + '_' + MODEL_NAME +  '_' + DATA_NAME
	## load data
	train_data, val_data = load_dataset_mrqa(DATA_NAME.lower())
else:
	init_pool = NUM_INIT_LB
	setting = 'regular'
	## set dir
	strategy_model_dir = MODEL_DIR + '/' + str(NUM_INIT_LB) + '_' + str(args_input.quota) + '_' + STRATEGY_NAME + '_' + MODEL_NAME +  '_' + DATA_NAME
	## load data
	squad = load_dataset(DATA_NAME.lower(), cache_dir=CACHE_DIR)
	if args_input.dev_mode:
		print('Use 4000 training data and 1500 testing data.')
		train_data = squad["train"].select(range(4000))
		val_data = squad["validation"]
	else:
		train_data = squad["train"]
		val_data = squad["validation"]
		# print('Use full training data and full testing data.')

## disable_caching
disable_caching()

## preprocess data
train_dataset, train_features, val_dataset, val_features = preprocess_data(train_data, val_data)
context_dict = get_context_id(train_data)

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
all_acc_dict = {}
all_acc_em_dict = {}
acq_time = []

begin = datetime.datetime.now()

res = {'exp': EXP_ID,
	   'setting': setting,
	   'ALStrategy': STRATEGY_NAME,
	   'dataset': DATA_NAME,
	   'model': MODEL_NAME,
	   'unlabeledPool': len(train_dataset) - init_pool,
	   'testingPool': len(val_dataset),
	   'initLabeledPool': init_pool,
	   'queryQuota': args_input.quota,
	   'queryBatchSize': NUM_QUERY,
	   'expRound': EXPE_ROUND,
	   'learningRate': LEARNING_RATE,
	   'trainingBatchSize': MODEL_BATCH,
	   'trainingEpoch': NUM_TRAIN_EPOCH,
	   'devMode': args_input.dev_mode,
	   'uniqueContex': UNIQ_CONTEXT,
	   'time': {
		   'start': str(begin),
	   }
	}

print('\nThe detail of this experiment:', json.dumps(res, indent=4))
print('\n')

## repeate experiment trials
while (EXPE_ROUND > 0): 
	EXPE_ROUND = EXPE_ROUND - 1
	print('Exp_round_{} start.'.format(args_input.expe_round - EXPE_ROUND))
	
	start = datetime.datetime.now()

	## record acc performance 
	acc = np.zeros(ITERATION + 1) # quota/batch runs + iter_0
	acc_em = np.zeros(ITERATION + 1)

	## generate initial labeled pool
	n_pool = len(train_dataset)
	labeled_idxs = np.zeros(n_pool, dtype=bool)

	if LOW_RES:
		save_model(device, pretrain_model_dir, strategy_model_dir)
	else:
		tmp_idxs = np.arange(n_pool)
		np.random.shuffle(tmp_idxs)
		
		if UNIQ_CONTEXT:
			tmp_idxs = tmp_idxs[:NUM_INIT_LB+extra]
			uc_tmp_idxs = get_unique_context(tmp_idxs, train_features, context_dict) # len() = almost num_query + extra
			iter_0_labeled_idxs = get_unique_sample(labeled_idxs, uc_tmp_idxs, n_pool, train_features)
			c_id = get_final_c_id(iter_0_labeled_idxs, train_features, context_dict) # len() = num_query
		else:
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

		## iteration 0 accuracy
		to_train(NUM_TRAIN_EPOCH, train_dataloader, device, model, optimizer, lr_scheduler)

	## load the selected validation data to DataLoader
	eval_dataloader = DataLoader(
		val_dataset, 
		collate_fn=default_data_collator, 
		batch_size=MODEL_BATCH
	)

	acc_scores_0 = get_pred(eval_dataloader, device, val_features, val_data) # add i=1 to use model from models_dir
	acc[0] = round(acc_scores_0['f1'], 4)
	acc_em[0] = round(acc_scores_0['exact_match'], 4)

	print('\nIterantion 0 done.')
	print('Testing accuracy: {}'.format(acc[0]))
	print('Testing accuracy em {}'.format(acc_em[0]))

	time = datetime.datetime.now()
	print('Time spent: {}\n'.format(time - start))
	
	## iteration 1 to i
	for i in range(1, ITERATION+1):
		print('Iteraion {} in exp_round_{} start.'.format(i, args_input.expe_round - EXPE_ROUND))

		## use total_query (NUM_QUERY + extra) to query instead of just NUM_QUERY
		total_query = NUM_QUERY + extra
		
		## query
		if STRATEGY_NAME == 'RandomSampling':
			q_idxs = random_sampling(labeled_idxs, total_query)
		elif STRATEGY_NAME == 'MarginSampling':
			q_idxs = margin(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, total_query)
		elif STRATEGY_NAME == 'LeastConfidence':
			q_idxs = least_confidence(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, total_query)
		elif STRATEGY_NAME == 'EntropySampling':
			q_idxs = entropy(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, total_query)
		elif STRATEGY_NAME == 'MarginSamplingDropout':
			q_idxs = margin_dropout(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, total_query)
		elif STRATEGY_NAME == 'LeastConfidenceDropout':
			q_idxs = least_confidence_dropout(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, total_query)
		elif STRATEGY_NAME == 'EntropySamplingDropout':
			q_idxs = entropy_dropout(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, total_query)
		elif STRATEGY_NAME == 'VarRatio':
			q_idxs = var_ratio(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, total_query)
		elif STRATEGY_NAME == 'BALDDropout':
			q_idxs = bald(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, total_query)
		elif STRATEGY_NAME == 'BatchBALDDropout':
			q_idxs = batch_bald(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, total_query)
		elif STRATEGY_NAME == 'MeanSTD':
			q_idxs = mean_std(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, total_query)
		elif STRATEGY_NAME == 'KMeansSampling':
			q_idxs = kmeans(n_pool, labeled_idxs, train_dataset, device, total_query)
		elif STRATEGY_NAME == 'KCenterGreedy':
			q_idxs = kcenter(n_pool, labeled_idxs, train_dataset, device, total_query)
		elif STRATEGY_NAME == 'BadgeSampling':
			q_idxs = badge(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, total_query)
		else:
			raise NotImplementedError

		print('Time spent for querying: {}\n'.format(datetime.datetime.now() - time))
		time = datetime.datetime.now()

		if UNIQ_CONTEXT:
			# print('in uc setting')
			if LOW_RES:
				# print('in lr setting')
				# print('num of idxs:', len(q_idxs))
				uc_q_idxs = get_unique_context(q_idxs, train_features, context_dict)
				# print('num of uc idxs:', len(uc_q_idxs))
			else:
				# print('not in lr setting')
				# print('num of idxs:', len(q_idxs))
				uc_q_idxs = get_unique_context(q_idxs, train_features, context_dict, c_id)
				# print('num of ucidxs:', len(uc_q_idxs))
			
			iter_i_labeled_idxs = get_unique_sample(labeled_idxs, uc_q_idxs, n_pool, train_features, i)
			c_id = get_final_c_id(iter_i_labeled_idxs, train_features, context_dict)
		else:
			# print('not in uc setting')
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
		print('\nIter_{} get_pred.'.format(i))
		acc_scores_i = get_pred(eval_dataloader, device, val_features, val_data)
		acc[i] = round(acc_scores_i['f1'], 4)
		acc_em[i] = round(acc_scores_i['exact_match'], 4)
		print('Iterantion {} done.'.format(i))
		print('Testing accuracy: {}'.format(acc[i]))
		print('Testing accuracy em: {}'.format(acc_em[i]))
		print('Time spent: {}\n'.format(datetime.datetime.now() - time))
		time = datetime.datetime.now()
		torch.cuda.empty_cache()
	
	## print results
	# print('SEED {}'.format(SEED))
	# print(STRATEGY_NAME)
	print('Exp_Round_{} done.'.format(args_input.expe_round - EXPE_ROUND))
	print('ExpRound_{} testing accuracy: {}'.format(args_input.expe_round - EXPE_ROUND, acc))
	print('ExpRound_{} testing accuracy em: {}\n'.format(args_input.expe_round - EXPE_ROUND, acc_em))
	all_acc.append(acc)
	all_acc_dict['ExpRound_' + str(args_input.expe_round - EXPE_ROUND)] = acc.tolist()
	all_acc_em_dict['ExpRound_' + str(args_input.expe_round - EXPE_ROUND)] = acc_em.tolist()
	
	## record acq time
	timestamp = re.sub('\.[0-9]*', '_', str(datetime.datetime.now())).replace(" ", "_").replace("-", "").replace(":","")
	end = datetime.datetime.now()
	acq_time.append(round(float((end-start).seconds), 3))

# cal mean & standard deviation
total_time = datetime.datetime.now() - begin
print('Time spent in total:', total_time)
res['time']['end'] = str(end)
res['time']['total'] = str(total_time)
res['em'] = all_acc_em_dict
res['f1'] = all_acc_dict
acc_m = []
acc_m_dict = {}

## save result
avg_acc = np.mean(np.array(all_acc), axis=0)
stddev_i_acc = np.std(np.array(all_acc), axis=0)

mean_acc_dict = {}
std_acc_dict = {}
for i in range(len(avg_acc)):
	if LOW_RES:
		mean_acc_dict['labelTotal_' + str(i * NUM_QUERY)] = round(avg_acc[i], 4)
		std_acc_dict['labelTotal_' + str(i * NUM_QUERY)] = round(stddev_i_acc[i], 4)
		tmp = 'When the size of training set is ' + str(i * NUM_QUERY) + ', ' + 'average accuracy is ' + str(round(avg_acc[i], 4)) + ', ' + 'std dev is ' + str(round(stddev_i_acc[i], 4)) + '.'
	else:
		mean_acc_dict['labelTotal_' + str(NUM_INIT_LB + i * NUM_QUERY)] = round(avg_acc[i], 4)
		std_acc_dict['labelTotal_' + str(NUM_INIT_LB + i * NUM_QUERY)] = round(stddev_i_acc[i], 4)
		tmp = 'When the size of training set is ' + str(NUM_INIT_LB + i * NUM_QUERY) + ', ' + 'average accuracy is ' + str(round(avg_acc[i], 4)) + ', ' + 'std dev is ' + str(round(stddev_i_acc[i], 4)) + '.'
	print(tmp)
res['f1MeanByBatch'] = mean_acc_dict
res['stdMeanByBatch'] = std_acc_dict

for i in range(len(all_acc)):
	acc_m.append(get_aubc(args_input.quota, NUM_QUERY, all_acc[i]))
	acc_m_dict['ExpRound_' + str(i+1)] = float(get_aubc(args_input.quota, NUM_QUERY, all_acc[i]))
	print('The experiment round ' + str(i+1) + ': ' + str(acc_m_dict['ExpRound_' + str(i+1)]))
res['AUBC'] = acc_m_dict

mean_acc, std_acc = get_mean_stddev(acc_m)
mean_time, stddev_time = get_mean_stddev(acq_time)
res['f1mean'] = float(mean_acc) 
res['std'] = float(std_acc)
res['time']['mean'] = str(datetime.timedelta(seconds=mean_time))
res['time']['std'] = str(datetime.timedelta(seconds=stddev_time))
print('\nmean acc: ' + str(res['f1mean']) + '. std dev acc: ' + str(res['std']))
print('mean time: ' + res['time']['mean'] + '. std dev acc: ' + res['time']['std'] + '\n')

with open(OUTPUT_DIR, "w") as fw:
	json.dump(res, fw, indent=4)
