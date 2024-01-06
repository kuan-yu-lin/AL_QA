from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from adapters import UniPELTConfig, AutoAdapterModel
import adapters
import numpy as np
import sys
import os

import arguments
from preprocess import preprocess_training_examples, preprocess_training_features, preprocess_validation_examples
from strategies import random_sampling, margin, least_confidence, entropy, margin_dropout, least_confidence_dropout, entropy_dropout, kcenter, kmeans, mean_std, bald, badge, batch_bald

CACHE_DIR =  os.path.abspath('') + '/.cache'
args_input = arguments.get_args()
LOW_RES = args_input.low_res
STRATEGY_NAME = args_input.ALstrategy

class Logger(object):
	def __init__(self, filename="Default.log"):
		self.terminal = sys.stdout
		self.log = open(filename, "a")

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)

	def flush(self):
		pass

def get_aubc(quota, bsize, resseq):
	# it is equal to use np.trapz for calculation
	ressum = 0.0
	if quota % bsize == 0:
		for i in range(len(resseq)-1):
			ressum = ressum + (resseq[i+1] + resseq[i]) * bsize / 2

	else:
		for i in range(len(resseq)-2):
			ressum = ressum + (resseq[i+1] + resseq[i]) * bsize / 2
		k = quota % bsize
		ressum = ressum + ((resseq[-1] + resseq[-2]) * k / 2)
	ressum = round(ressum / quota,3)
	
	return ressum

def get_mean_stddev(datax):
	return round(np.mean(datax),4),round(np.std(datax),4)

def get_model(m):
	if m.lower() == 'bert':
		return 'bert-base-uncased'
	elif m.lower() == 'bertlarge':
		return 'bert-large-uncased'
	elif m.lower() == 'roberta':
		return 'roberta-base'
	elif m.lower() == 'robertalarge':
		return 'roberta-large'

def preprocess_data(train_data, val_data):
	tokenizer = AutoTokenizer.from_pretrained(get_model(args_input.model))

	train_dataset = train_data.map(
		preprocess_training_examples,
		batched=True,
		remove_columns=train_data.column_names,
		fn_kwargs=dict(tokenizer=tokenizer)
	)
	train_features = train_data.map(
		preprocess_training_features,
		batched=True,
		remove_columns=train_data.column_names,
		fn_kwargs=dict(tokenizer=tokenizer)
	)
	val_dataset = val_data.map(
		preprocess_validation_examples,
		batched=True,
		remove_columns=val_data.column_names,
		fn_kwargs=dict(tokenizer=tokenizer)
	)
	val_features = val_data.map(
		preprocess_validation_examples,
		batched=True,
		remove_columns=val_data.column_names,
		fn_kwargs=dict(tokenizer=tokenizer)
	)

	train_dataset.set_format("torch")
	train_features.set_format("torch")
	val_dataset = val_dataset.remove_columns(["offset_mapping"])
	val_dataset.set_format("torch")
	val_features.set_format("torch")

	return train_dataset, train_features, val_dataset, val_features

def load_dataset_mrqa(d):
	'''
	return train_set, val_set
	'''
	data = load_dataset("mrqa", cache_dir=CACHE_DIR)
	if d.lower() == 'squad':
		# the first to 86588th in train set
		# the first to 10507th in val set
		squad_train = data['train'].select(range(86588))
		squad_val = data['validation'].select(range(10507))
		for t in squad_train: assert t['subset'] == 'SQuAD', 'Please select corrrect train data for SQuAD.'
		for v in squad_val: assert v['subset'] == 'SQuAD', 'Please select corrrect validation data for SQuAD.'
		return squad_train, squad_val
	elif d.lower() == 'newsqa':
		# the 86589th to 160748th in train set
		# the 10508th to 14719th in val set
		data_set = data['train'].select(range(86588, 160748))
		newsqa_train = data_set.shuffle(1127).select(range(10000))
		newsqa_val = data['validation'].select(range(10507, 14719))
		for t in newsqa_train: assert t['subset'] == 'NewsQA', 'Please select corrrect train data for NewQA.'
		for v in newsqa_val: assert v['subset'] == 'NewsQA', 'Please select corrrect validation data for NewQA.'
		return newsqa_train, newsqa_val
	elif d.lower() == 'searchqa':
		# the 222437th to 339820th in train set
		# the 22505th to 39484th in val set
		data_set = data['train'].select(range(222436, 339820))
		searchqa_train = data_set.shuffle(1127).select(range(10000))
		searchqa_val = data['validation'].select(range(22504, 39484))	
		for t in searchqa_train: assert t['subset'] == 'SearchQA', 'Please select corrrect train data for SearchQA.'
		for v in searchqa_val: assert v['subset'] == 'SearchQA', 'Please select corrrect validation data for SearchQA.'
		return searchqa_train, searchqa_val
	elif d.lower() == 'naturalquestions':
		# the 412,749th to 516,819th in train set
		# the 45,389th to 58,224th in val set
		data_set = data['train'].select(range(412748, 516819))
		naturalquestions_train = data_set.shuffle(1127).select(range(10000))
		naturalquestions_val = data['validation'].select(range(45385, 58221))
		for t in naturalquestions_train: assert t['subset'] == 'NaturalQuestionsShort', 'Please select corrrect train data for NaturalQuestions.'
		for v in naturalquestions_val: assert v['subset'] == 'NaturalQuestionsShort', 'Please select corrrect validation data for NaturalQuestions.'
		return naturalquestions_train, naturalquestions_val
	elif d.lower() == 'bioasq':
		# the first to the 1504th in the test set
		sub = data['test'].select(range(1504))
		len_sub_val = len(sub) // 10
		bioasq_train = sub.select(range(len_sub_val, len(sub)))
		bioasq_val = sub.select(range(len_sub_val))
		for t in bioasq_train: assert t['subset'] == 'BioASQ', 'Please select corrrect train data for BioASQ.'
		for v in bioasq_val: assert v['subset'] == 'BioASQ', 'Please select corrrect validation data for BioASQ.'
		return bioasq_train, bioasq_val
	elif d.lower() == 'textbookqa':
		# the 8131st to 9633rd
		sub = data['test'].select(range(8130, 9633))
		len_sub_val = len(sub) // 10
		textbookqa_train = sub.select(range(len_sub_val, len(sub)))
		textbookqa_val = sub.select(range(len_sub_val)) 
		for t in textbookqa_train: assert t['subset'] == 'TextbookQA', 'Please select corrrect train data for TextbookQA.'
		for v in textbookqa_val: assert v['subset'] == 'TextbookQA', 'Please select corrrect validation data for TextbookQA.'
		return textbookqa_train, textbookqa_val
	elif d.lower() == 'drop': # Discrete Reasoning Over Paragraphs
		# the 1505th to 3007th in test set
		sub = data['test'].select(range(1504, 3007))
		len_sub_val = len(sub) // 10
		drop_train = sub.select(range(len_sub_val, len(sub)))
		drop_val = sub.select(range(len_sub_val))
		for t in drop_train: assert t['subset'] == 'DROP', 'Please select corrrect train data for DROP.'
		for v in drop_val: assert v['subset'] == 'DROP', 'Please select corrrect validation data for DROP.'
		return drop_train, drop_val

def query(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, i):
	if STRATEGY_NAME == 'RandomSampling':
		iter_i_labeled_idxs, ssi_ = random_sampling(n_pool, labeled_idxs, train_dataset, train_features, device, i)
	elif STRATEGY_NAME == 'MarginSampling':
		iter_i_labeled_idxs, ssi_ = margin(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, i)
	elif STRATEGY_NAME == 'LeastConfidence':
		iter_i_labeled_idxs, ssi_ = least_confidence(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, i)
	elif STRATEGY_NAME == 'EntropySampling':
		iter_i_labeled_idxs, ssi_ = entropy(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, i)
	elif STRATEGY_NAME == 'MarginSamplingDropout':
		iter_i_labeled_idxs, ssi_ = margin_dropout(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, i)
	elif STRATEGY_NAME == 'LeastConfidenceDropout':
		iter_i_labeled_idxs, ssi_ = least_confidence_dropout(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, i)
	elif STRATEGY_NAME == 'EntropySamplingDropout':
		iter_i_labeled_idxs, ssi_ = entropy_dropout(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, i)
	elif STRATEGY_NAME == 'BALDDropout':
		iter_i_labeled_idxs, ssi_ = bald(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, i)
	elif STRATEGY_NAME == 'BatchBALDDropout':
		iter_i_labeled_idxs, ssi_ = batch_bald(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, i)
	elif STRATEGY_NAME == 'MeanSTD':
		iter_i_labeled_idxs, ssi_ = mean_std(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, i)
	elif STRATEGY_NAME == 'KMeansSampling':
		iter_i_labeled_idxs, ssi_ = kmeans(n_pool, labeled_idxs, train_dataset, train_features, device, i)
	elif STRATEGY_NAME == 'KCenterGreedy':
		if LOW_RES and i == 1:
			iter_i_labeled_idxs, ssi_ = random_sampling(n_pool, labeled_idxs, train_dataset, train_features, device, i)
		else:
			iter_i_labeled_idxs, ssi_ = kcenter(n_pool, labeled_idxs, train_dataset, train_features, device, i)
	elif STRATEGY_NAME == 'BadgeSampling':
		iter_i_labeled_idxs, ssi_ = badge(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, i)
	else:
		raise NotImplementedError
	return iter_i_labeled_idxs, ssi_

def save_model(device, pretrain_dir, strategy_dir):
	'''
	Copy and save model from pretrain_models to current trained models.
	'''
	pretrain_model = AutoAdapterModel.from_pretrained(pretrain_dir).to(device)
	# pretrain_model = AutoModelForQuestionAnswering.from_pretrained(pretrain_dir).to(device)
	adapters.init(pretrain_model)
	config = UniPELTConfig()
	pretrain_model.add_adapter("unipelt", config=config)
	model_to_save = pretrain_model.module if hasattr(pretrain_model, 'module') else pretrain_model
	model_to_save.save_pretrained(strategy_dir)
