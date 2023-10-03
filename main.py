#######################
# test random to dropout
# delete shuffle in query
# 

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
from preprocess import preprocess_training_examples, preprocess_training_features, preprocess_validation_examples
from model import to_train, get_pred
from utils import get_model, Logger, get_aubc, get_mean_stddev, get_unique_sample, get_unique_sample_and_context
from query import (
    random_sampling_query, 
    margin_sampling_query, 
    least_confidence_query, 
    entropy_query,
    margin_sampling_dropout_query,
    least_confidence_dropout_query,
    entropy_dropout_query,
    var_ratio_query,
    bald_query,
    mean_std_query,
    kmeans_query,
    kcenter_greedy_query,
    badge_query
)

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
UNIQ_CONTEXT = args_input.unique_context

model_dir = '/mount/arbeitsdaten31/studenten1/linku/models'
strategy_model_dir = model_dir + '/' + str(NUM_INIT_LB) + '_' + str(args_input.quota) + '_' + STRATEGY_NAME + '_' + MODEL_NAME +  '_' + DATA_NAME

CACHE_DIR = '/mount/arbeitsdaten31/studenten1/linku/.cache'

## load data
squad = load_dataset(DATA_NAME.lower(), cache_dir=CACHE_DIR)
if args_input.toy_exp:
	print('Use 4000 training data and 1500 testing data.')
	squad["train"] = squad["train"].select(range(4000))
	squad["validation"] = squad["validation"].select(range(1500))
else:
	print('Use full training data and full testing data.')

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
train_features = squad["train"].map(
    preprocess_training_features,
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
train_features.set_format("torch")
val_dataset = val_dataset.remove_columns(["offset_mapping"])
val_dataset.set_format("torch")
val_features.set_format("torch")

# get the number of extra data after preprocessing
extra = min(NUM_QUERY, len(train_dataset) - len(squad['train']))

## seed
SEED = 1127
os.environ["CUDA_VISIBLE_DEVICES"] = str(args_input.gpu)

# fix random seed
np.random.seed(SEED)
torch.manual_seed(SEED)

## device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

	## generate initial labeled pool
	n_pool = len(train_dataset)
	labeled_idxs = np.zeros(n_pool, dtype=bool)

	tmp_idxs = np.arange(n_pool)
	np.random.shuffle(tmp_idxs)
	
	if UNIQ_CONTEXT:
		iter_0_labeled_idxs = get_unique_sample_and_context(labeled_idxs, tmp_idxs, n_pool, train_features)
	else:
		iter_0_labeled_idxs = get_unique_sample(labeled_idxs, tmp_idxs, n_pool, train_features)
	
	# difference_0 = 0
	# num_set_ex_id_0 = 0

	# while num_set_ex_id_0 != NUM_INIT_LB:        
	# 	labeled_idxs[tmp_idxs[:NUM_INIT_LB + difference_0]] = True
	# 	iter_0_labeled_idxs = np.arange(n_pool)[labeled_idxs]

	# 	iter_0_samples = train_features.select(indices=iter_0_labeled_idxs)
	# 	num_set_ex_id_0 = len(set(iter_0_samples['example_id']))

	# 	difference_0 = NUM_INIT_LB - num_set_ex_id_0

	## record acc performance 
	acc = np.zeros(ITERATION + 1) # quota/batch runs + iter_0
	acc_em = np.zeros(ITERATION + 1)

	## load the selected train data to DataLoader
	train_dataloader = DataLoader(
		train_dataset.select(indices=iter_0_labeled_idxs),
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
	
	acc_scores_0 = get_pred(eval_dataloader, device, val_features, squad['validation']) # add i=1 to use model from models_dir
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
			q_idxs = margin_sampling_query(n_pool, labeled_idxs, train_dataset, train_features, squad['train'], device, total_query)
		elif STRATEGY_NAME == 'LeastConfidence':
			q_idxs = least_confidence_query(n_pool, labeled_idxs, train_dataset, train_features, squad['train'], device, total_query)
		elif STRATEGY_NAME == 'EntropySampling':
			q_idxs = entropy_query(n_pool, labeled_idxs, train_dataset, train_features, squad['train'], device, total_query)
		elif STRATEGY_NAME == 'MarginSamplingDropout':
			q_idxs = margin_sampling_dropout_query(n_pool, labeled_idxs, train_dataset, train_features, squad['train'], device, total_query)
		elif STRATEGY_NAME == 'LeastConfidenceDropout':
			q_idxs = least_confidence_dropout_query(n_pool, labeled_idxs, train_dataset, train_features, squad['train'], device, total_query)
		elif STRATEGY_NAME == 'EntropySamplingDropout':
			q_idxs = entropy_dropout_query(n_pool, labeled_idxs, train_dataset, train_features, squad['train'], device, total_query)
		elif STRATEGY_NAME == 'VarRatio':
			q_idxs = var_ratio_query(n_pool, labeled_idxs, train_dataset, train_features, squad['train'], device, total_query)
		elif STRATEGY_NAME == 'BALDDropout':
			q_idxs = bald_query(n_pool, labeled_idxs, train_dataset, train_features, squad['train'], device, total_query)
		elif STRATEGY_NAME == 'MeanSTD':
			q_idxs = mean_std_query(n_pool, labeled_idxs, train_dataset, train_features, squad['train'], device, total_query)
		elif STRATEGY_NAME == 'KMeansSampling':
			q_idxs = kmeans_query(n_pool, labeled_idxs, train_dataset, device, total_query)
		elif STRATEGY_NAME == 'KCenterGreedy':
			q_idxs = kcenter_greedy_query(n_pool, labeled_idxs, train_dataset, device, total_query)
		elif STRATEGY_NAME == 'BadgeSampling':
			q_idxs = badge_query(n_pool, labeled_idxs, train_dataset, train_features, squad['train'], device, total_query)
		else:
			raise NotImplementedError

		print('Time spent for querying:', (datetime.datetime.now() - time))
		time = datetime.datetime.now()

		## update
		if UNIQ_CONTEXT:
			iter_i_labeled_idxs = get_unique_sample_and_context(labeled_idxs, q_idxs, n_pool, train_features, i)
		else:
			iter_i_labeled_idxs = get_unique_sample(labeled_idxs, q_idxs, n_pool, train_features, i)
		 
		# ## goal of total query data: sum NUM_QUERY and the number of set iter_0_data
		# num_set_query_i = NUM_QUERY * i + NUM_INIT_LB
		
		# difference_i = 0
		# num_set_ex_id_i = 0

		# while num_set_ex_id_i != num_set_query_i:        
		# 	labeled_idxs[q_idxs[:NUM_QUERY + difference_i]] = True
		# 	iter_i_labeled_idxs = np.arange(n_pool)[labeled_idxs]

		# 	iter_i_samples = train_features.select(indices=iter_i_labeled_idxs)
		# 	num_set_ex_id_i = len(set(iter_i_samples['example_id']))

		# 	difference_i = num_set_query_i - num_set_ex_id_i

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
		acc_scores_i = get_pred(eval_dataloader, device, val_features, squad['validation'])
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
file_name_res = str(NUM_INIT_LB) + '_' + str(args_input.quota) + '_' + STRATEGY_NAME + '_' + MODEL_NAME + '_' + DATA_NAME + '_normal_res.txt'
file_res =  open(os.path.join(os.path.abspath('') + '/results', '%s' % file_name_res),'w')

file_res.writelines('dataset: {}'.format(DATA_NAME) + '\n')
file_res.writelines('AL strategy: {}'.format(STRATEGY_NAME) + '\n')
file_res.writelines('number of labeled pool: {}'.format(NUM_INIT_LB) + '\n')
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
	tmp = 'When the size of training set is ' + str(NUM_INIT_LB + i * NUM_QUERY) + ', ' + 'average accuracy is ' + str(round(avg_acc[i], 4)) + ', ' + 'std dev is ' + str(round(stddev_i_acc[i], 4)) + '.' + '\n'
	file_res.writelines(tmp)

file_res.writelines('mean acc: ' + str(mean_acc) + '. std dev acc: ' + str(stddev_acc) + '\n')
file_res.writelines('mean time: ' + str(datetime.timedelta(seconds=mean_time)) + '. std dev acc: ' + str(datetime.timedelta(seconds=stddev_time)) + '\n')

file_res.close()