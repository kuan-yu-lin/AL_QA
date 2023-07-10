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

import warnings
import sys
import os
import re
import datetime

import arguments
from preprocess import *
from model import *
from utils import *
from query import *

CACHE_DIR = '/mount/arbeitsdaten31/studenten1/linku/cache'
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_MODULES_CACHE'] = CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = CACHE_DIR

args_input = arguments.get_args()
NUM_QUERY = args_input.batch # my_query_test: 35 # deepAL+: 128
NUM_INIT_LB = args_input.initseed # my_query_test: 100 # deepAL+: 1000
NUM_ROUND = int(args_input.quota / args_input.batch) # my_query_test: 100 / 35 # deepAL+: 1000 / 128
DATA_NAME = args_input.dataset_name # my_query_test: 'SQuAD'
STRATEGY_NAME = args_input.ALstrategy

model_dir = '/mount/arbeitsdaten31/studenten1/linku/models'

## load data
squad = load_dataset(DATA_NAME.lower())
squad["train"] = squad["train"].select(range(4000))
squad["validation"] = squad["validation"].select(range(1500))

## preprocess data
train_dataset = squad["train"].map(
    preprocess_training_examples,
    batched=True,
    remove_columns=squad["train"].column_names,
)
train_features = squad["train"].map(
    preprocess_training_features,
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
train_features.set_format("torch")
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

sys.stdout = Logger(os.path.abspath('') + '/logfile/' + DATA_NAME+ '_'  + STRATEGY_NAME + '_' + str(NUM_QUERY) + '_' + str(NUM_INIT_LB) +  '_' + str(args_input.quota) + '_normal_log.txt')
warnings.filterwarnings('ignore')

## start experiment
iteration = args_input.iteration
model_batch = args_input.model_batch
NUM_TRAIN_EPOCH = args_input.train_epochs

all_acc = []
acq_time = []

# repeate # iteration trials
while (iteration > 0): 
	iteration = iteration - 1

	start = datetime.datetime.now()

	## generate initial labeled pool
	n_pool = len(train_dataset)
	labeled_idxs = np.zeros(n_pool, dtype=bool)

	tmp_idxs = np.arange(n_pool)
	np.random.shuffle(tmp_idxs)
	labeled_idxs[tmp_idxs[:NUM_INIT_LB]] = True

	run_0_labeled_idxs = np.arange(n_pool)[labeled_idxs]

	## record acc performance 
	acc = np.zeros(NUM_ROUND + 1) # quota/batch runs + run_0
	acc_em = np.zeros(NUM_ROUND + 1)

	## load the selected data to DataLoader
	eval_dataloader = DataLoader(
		val_dataset, 
		collate_fn=default_data_collator, 
		batch_size=8
	)

	## print info
	print(DATA_NAME)
	print(STRATEGY_NAME)
	
	## round 0 accuracy
	acc[0] = get_pred(eval_dataloader, device, val_features, squad['validation'], rd=0)['f1']
	# print(acc) # 77.96450701

	print('Round 0\ntesting accuracy {}'.format(acc[0]))
	print('\n')
	
	## round 1 to rd
	for rd in range(1, NUM_ROUND+1):
		print('Round {}'.format(rd))

		## query
		if STRATEGY_NAME == 'RandomSampling':
			q_idxs = random_sampling_query(labeled_idxs, NUM_QUERY)
		elif STRATEGY_NAME == 'MarginSampling':
			q_idxs = margin_sampling_query(n_pool, labeled_idxs, train_dataset, train_features, squad['train'], device, NUM_QUERY, rd)
		elif STRATEGY_NAME == 'LeastConfidence':
			q_idxs = least_confidence_query(n_pool, labeled_idxs, train_dataset, train_features, squad['train'], device, NUM_QUERY)
		elif STRATEGY_NAME == 'EntropySampling':
			q_idxs = entropy_query(n_pool, labeled_idxs, train_dataset, train_features, squad['train'], device, NUM_QUERY)
		elif STRATEGY_NAME == 'MarginSamplingDropout':
			q_idxs = margin_sampling_dropout_query(n_pool, labeled_idxs, train_dataset, train_features, squad['train'], device, NUM_QUERY)
		elif STRATEGY_NAME == 'LeastConfidenceDropout':
			q_idxs = least_confidence_dropout_query(n_pool, labeled_idxs, train_dataset, train_features, squad['train'], device, NUM_QUERY)
		elif STRATEGY_NAME == 'EntropySamplingDropout':
			q_idxs = entropy_dropout_query(n_pool, labeled_idxs, train_dataset, train_features, squad['train'], device, NUM_QUERY)
		elif STRATEGY_NAME == 'VarRatio':
			q_idxs = var_ratio_query(n_pool, labeled_idxs, train_dataset, train_features, squad['train'], device, NUM_QUERY)
		elif STRATEGY_NAME == 'BALDDropout':
			q_idxs = bayesian_query(n_pool, labeled_idxs, train_dataset, train_features, squad['train'], device, NUM_QUERY)
		elif STRATEGY_NAME == 'MeanSTD':
			q_idxs = mean_std_query(n_pool, labeled_idxs, train_dataset, train_features, squad['train'], device, NUM_QUERY)
		elif STRATEGY_NAME == 'KMeansSampling':
			q_idxs = kmeans_query(n_pool, labeled_idxs, train_dataset, train_features, squad['train'], device, NUM_QUERY, rd)
		elif STRATEGY_NAME == 'KCenterGreedy':
			q_idxs = kcenter_greedy_query(n_pool, labeled_idxs, train_dataset, train_features, squad['train'], device, NUM_QUERY, rd)
		elif STRATEGY_NAME == 'KCenterGreedyPCA': # not sure
			q_idxs = kcenter_greedy_PCA_query((n_pool, labeled_idxs, train_dataset, train_features, squad['train'], device, NUM_QUERY, rd))
		# elif STRATEGY_NAME == 'BadgeSampling':
		# 	q_idxs = badge_query()
		# elif STRATEGY_NAME == 'LossPredictionLoss':
		# 	# different net!
		# 	q_idxs = loss_prediction_query()
		# elif STRATEGY_NAME == 'CEALSampling':
		# 	# why use 'CEALSampling' in STRATEGY_NAME
		# 	q_idxs = ceal_query()
		else:
			raise NotImplementedError

		## update
		labeled_idxs[q_idxs] = True
		run_rd_labeled_idxs = np.arange(n_pool)[labeled_idxs]

		train_dataloader_rd = DataLoader(
			train_dataset.select(indices=run_rd_labeled_idxs),
			shuffle=True,
			collate_fn=default_data_collator,
			batch_size=8,
		)

		num_update_steps_per_epoch_rd = len(train_dataloader_rd)
		num_training_steps_rd = NUM_TRAIN_EPOCH * num_update_steps_per_epoch_rd

		if rd == 1:
			config = BertConfig.from_pretrained(pretrain_model_dir)
		else:
			config = BertConfig.from_pretrained(model_dir)
		model_rd = AutoModelForQuestionAnswering.from_config(config).to(device)
		optimizer_rd = AdamW(model_rd.parameters(), lr=1e-4)

		lr_scheduler_rd = get_scheduler(
			"linear",
			optimizer=optimizer_rd,
			num_warmup_steps=0,
			num_training_steps=num_training_steps_rd,
		)

		## train
		to_train(NUM_TRAIN_EPOCH, train_dataloader_rd, device, model_rd, optimizer_rd, lr_scheduler_rd)

		## round rd accuracy
		acc[rd] = get_pred(eval_dataloader, device, val_features, squad['validation'], rd)['f1']
		print('testing accuracy {}'.format(acc[rd]))
		print('\n')

		torch.cuda.empty_cache()
	
	## print results
	print('SEED {}'.format(SEED))
	print(STRATEGY_NAME)
	print(acc)
	all_acc.append(acc)
	
	## save model and record acq time
	timestamp = re.sub('\.[0-9]*','_',str(datetime.datetime.now())).replace(" ", "_").replace("-", "").replace(":","")
	final_model_dir = model_dir + '/' + timestamp + DATA_NAME+ '_'  + STRATEGY_NAME + '_' + str(NUM_QUERY) + '_' + str(NUM_INIT_LB) +  '_' + str(args_input.quota)
	os.makedirs(final_model_dir, exist_ok=True)
	end = datetime.datetime.now()
	acq_time.append(round(float((end-start).seconds), 3))

	final_model = AutoModelForQuestionAnswering.from_pretrained(model_dir).to(device)
	model_to_save = final_model.module if hasattr(final_model, 'module') else final_model 
	model_to_save.save_pretrained(final_model_dir)

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
