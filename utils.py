from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import numpy as np
import sys
import os

import arguments
from preprocess import preprocess_training_examples, preprocess_training_features, preprocess_validation_examples, preprocess_training_examples_lowRes, preprocess_training_features_lowRes, preprocess_validation_examples_lowRes
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

CACHE_DIR =  os.path.abspath('') + '/.cache'

def decode_id():
	args_input = arguments.get_args()
	[p1, p2, p3, p4] = list(str(args_input.exp_id))[:4]
	UNIQ_CONTEXT = False
	LOW_RES = False
	MODEL_NAME = 'RoBERTa'

	if p1 == '2':
		MODEL_NAME = 'BERT'
	elif p1 == '3':
		MODEL_NAME = 'RoBERTaLarge'
	elif p1 == '4':
		MODEL_NAME = 'BERTLarge'
	elif p1 == '5':
		UNIQ_CONTEXT = True

	if p2 == '2': LOW_RES = True
	
	if p3 == '1':
		DATA_NAME = 'SQuAD'
	elif p3 == '2':
		DATA_NAME = 'BioASQ'
	elif p3 == '3':
		DATA_NAME = 'DROP'
	elif p3 == '4':
		DATA_NAME = 'TextbookQA'
	elif p3 == '5':
		DATA_NAME = 'NewsQA'
	elif p3 == '6':
		DATA_NAME = 'SearchQA'
	elif p3 == '7':
		DATA_NAME = 'NaturalQuestions'
	
	if p4 == 'a':
		STRATEGY_NAME = 'RandomSampling'
	elif p4 == 'b':
		STRATEGY_NAME = 'MarginSampling'
	elif p4 == 'c':
		STRATEGY_NAME = 'LeastConfidence'
	elif p4 == 'd':
		STRATEGY_NAME = 'EntropySampling'
	elif p4 == 'e':
		STRATEGY_NAME = 'MarginSamplingDropout'
	elif p4 == 'f':
		STRATEGY_NAME = 'LeastConfidenceDropout'
	elif p4 == 'g':
		STRATEGY_NAME = 'EntropySamplingDropout'
	elif p4 == 'h':
		STRATEGY_NAME = 'KMeansSampling'
	elif p4 == 'i':
		STRATEGY_NAME = 'KCenterGreedy'
	elif p4 == 'j':
		STRATEGY_NAME = 'MeanSTD'
	elif p4 == 'k':
		STRATEGY_NAME = 'BALDDropout'
	elif p4 == 'l':
		STRATEGY_NAME = 'BadgeSampling'
	elif p4 == 'm':
		STRATEGY_NAME = 'BatchBALD'
	
	return LOW_RES, DATA_NAME, STRATEGY_NAME, MODEL_NAME, UNIQ_CONTEXT

LOW_RES, DATA_NAME, STRATEGY_NAME, MODEL_NAME, UNIQ_CONTEXT = decode_id()

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
	tokenizer = AutoTokenizer.from_pretrained(get_model(MODEL_NAME))

	if LOW_RES:
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
	else:
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
		iter_i_labeled_idxs = random_sampling(n_pool, labeled_idxs, train_features, i)
	elif STRATEGY_NAME == 'MarginSampling':
		iter_i_labeled_idxs = margin(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, i)
	elif STRATEGY_NAME == 'LeastConfidence':
		iter_i_labeled_idxs = least_confidence(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, i)
	elif STRATEGY_NAME == 'EntropySampling':
		iter_i_labeled_idxs = entropy(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, i)
	elif STRATEGY_NAME == 'MarginSamplingDropout':
		iter_i_labeled_idxs = margin_dropout(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, i)
	elif STRATEGY_NAME == 'LeastConfidenceDropout':
		iter_i_labeled_idxs = least_confidence_dropout(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, i)
	elif STRATEGY_NAME == 'EntropySamplingDropout':
		iter_i_labeled_idxs = entropy_dropout(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, i)
	elif STRATEGY_NAME == 'BALDDropout':
		iter_i_labeled_idxs = bald(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, i)
	elif STRATEGY_NAME == 'BatchBALDDropout':
		iter_i_labeled_idxs = batch_bald(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, i)
	elif STRATEGY_NAME == 'MeanSTD':
		iter_i_labeled_idxs = mean_std(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, i)
	elif STRATEGY_NAME == 'KMeansSampling':
		iter_i_labeled_idxs = kmeans(n_pool, labeled_idxs, train_dataset, train_features, device, i)
	elif STRATEGY_NAME == 'KCenterGreedy':
		if LOW_RES and i == 1:
			iter_i_labeled_idxs = random_sampling(n_pool, labeled_idxs, train_features, i)
		else:
			iter_i_labeled_idxs = kcenter(n_pool, labeled_idxs, train_dataset, train_features, device, i)
	elif STRATEGY_NAME == 'BadgeSampling':
		iter_i_labeled_idxs = badge(n_pool, labeled_idxs, train_dataset, train_features, train_data, device, i)
	else:
		raise NotImplementedError
	return iter_i_labeled_idxs

def save_model(device, pretrain_dir, strategy_dir):
    '''
    Copy and save model from pretrain_models to current trained models.
    '''
    pretrain_model = AutoModelForQuestionAnswering.from_pretrained(pretrain_dir).to(device)
    model_to_save = pretrain_model.module if hasattr(pretrain_model, 'module') else pretrain_model 
    model_to_save.save_pretrained(strategy_dir)



	