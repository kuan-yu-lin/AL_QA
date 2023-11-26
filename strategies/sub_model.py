from collections import defaultdict
from tqdm.auto import tqdm
import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering
from adapters import AutoAdapterModel
from torch.autograd import Variable
import torch.nn.functional as F
from copy import deepcopy
import os
import sys
sys.path.insert(0, './')

import arguments
args_input = arguments.get_args()
EXP_ID = str(args_input.exp_id)
MODEL_NAME = args_input.model
LOW_RES = args_input.low_res
DATA_NAME = args_input.dataset
STRATEGY_NAME = args_input.ALstrategy
NUM_QUERY = args_input.batch
NUM_INIT_LB = args_input.initseed

if args_input.dev_mode:
	MODEL_DIR = os.path.abspath('') + '/dev_models'
else:
	MODEL_DIR = os.path.abspath('') + '/models'

strategy_model_dir = MODEL_DIR + '/' + EXP_ID

def get_prob(dataloader, device, features, examples):
	if LOW_RES:
		model = AutoAdapterModel.from_pretrained(strategy_model_dir).to(device)
		model.train_adapter("unipelt") # activate adapter
	else:
		model = AutoModelForQuestionAnswering.from_pretrained(strategy_model_dir).to(device)

	model.eval()
	start_logits = []
	end_logits = []

	for batch in tqdm(dataloader, desc="Evaluating_prob"):
		batch = {key: value.to(device) for key, value in batch.items()}
		with torch.no_grad():
			outputs = model(**batch)

		start_logits.append(outputs.start_logits.cpu().numpy())
		end_logits.append(outputs.end_logits.cpu().numpy())

	start_logits = np.concatenate(start_logits)
	end_logits = np.concatenate(end_logits)
	start_logits = start_logits[: len(features)]
	end_logits = end_logits[: len(features)]

	prob_dict = {}
	example_to_features = defaultdict(list)
	max_answer_length = 30
	n_best = 20
	
	for idx, feature in enumerate(features):
		example_to_features[feature["example_id"]].append(idx)

	for example in tqdm(examples):
		# TODO: use squad in mrqa
		example_id = example["qid"]
		answers = []

		# Loop through all features associated with that example
		for feature_index in example_to_features[example_id]:
			start_logit = start_logits[feature_index]
			end_logit = end_logits[feature_index]
			offsets = features[feature_index]["offset_mapping"]

			start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
			end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
			for start_index in start_indexes:
				for end_index in end_indexes:
					# Skip answers that are not fully in the context
					if offsets[start_index] is None or offsets[end_index] is None:
						continue
					# Skip answers with a length that is either < 0 or > max_answer_length
					if (
						end_index < start_index
						or end_index - start_index + 1 > max_answer_length
					):
						continue

					answers.append(start_logit[start_index] + end_logit[end_index])
		
			if len(answers) > 1:
				prob_dict[feature_index] = softmax(answers)
			elif example_to_features[example_id] != []:
				prob_dict[feature_index] = np.array([0])
	
	return prob_dict

def get_prob_dropout(dataloader, device, features, examples, n_drop=10):
	if LOW_RES:
		model = AutoAdapterModel.from_pretrained(strategy_model_dir).to(device)
		model.train_adapter("unipelt") # activate adapter
	else:
		model = AutoModelForQuestionAnswering.from_pretrained(strategy_model_dir).to(device)
	
	model.train()
	prob_dict = {}
	
	for i in range(n_drop):
		start_logits = []
		end_logits = []
		for batch in tqdm(dataloader, desc="Evaluating_prob_dropout"):
			batch = {key: value.to(device) for key, value in batch.items()}
			with torch.no_grad():
				outputs = model(**batch)

			start_logits.append(outputs.start_logits.cpu().numpy())
			end_logits.append(outputs.end_logits.cpu().numpy())

		start_logits = np.concatenate(start_logits)
		end_logits = np.concatenate(end_logits)
		start_logits = start_logits[: len(features)]
		end_logits = end_logits[: len(features)]

		example_to_features = defaultdict(list)
		max_answer_length = 30
		n_best = 20
			
		for idx, feature in enumerate(features):
			example_to_features[feature["example_id"]].append(idx)

		n = 0
		for example in tqdm(examples):
			# TODO: use squad in mrqa
			example_id = example["qid"]
			answers = []

			# Loop through all features associated with that example
			for feature_index in example_to_features[example_id]:
				start_logit = start_logits[feature_index]
				end_logit = end_logits[feature_index]
				offsets = features[feature_index]["offset_mapping"]

				start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
				end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
				for start_index in start_indexes:
					for end_index in end_indexes:
						# Skip answers that are not fully in the context
						if offsets[start_index] is None or offsets[end_index] is None:
							continue
						# Skip answers with a length that is either < 0 or > max_answer_length
						if (
							end_index < start_index
							or end_index - start_index + 1 > max_answer_length
						):
							continue

						answers.append(start_logit[start_index] + end_logit[end_index])

			if 1 < len(answers) < 200: # pad to same numbers of possible answers
				zero_list = [0] * (200 - len(answers))
				answers.extend(zero_list)
			elif len(answers) >= 200:
				answers = answers[:200]

			if len(answers) > 1:
				if example_to_features[example_id][0] not in prob_dict:
					prob_dict[example_to_features[example_id][0]] = softmax(answers)
				else:
					prob_dict[example_to_features[example_id][0]] += softmax(answers)
			elif example_to_features[example_id] != []:
				if example_to_features[example_id][0] not in prob_dict:
					prob_dict[example_to_features[example_id][0]] = np.array([0])   

	for key in prob_dict.keys():
		prob_dict[key] /= n_drop

	return prob_dict

def get_prob_dropout_split(dataloader, device, features, examples, n_drop=10):
	## use tensor to save the answers

	if LOW_RES:
		model = AutoAdapterModel.from_pretrained(strategy_model_dir).to(device)
		model.train_adapter("unipelt") # activate adapter
	else:
		model = AutoModelForQuestionAnswering.from_pretrained(strategy_model_dir).to(device)
	
	model.train()

	probs = torch.zeros([n_drop, len(dataloader.dataset), 200])
	
	for i in range(n_drop):
		start_logits = []
		end_logits = []
		for batch in tqdm(dataloader, desc="Evaluating_prob_dropout"):
			batch = {key: value.to(device) for key, value in batch.items()}
			with torch.no_grad():
				outputs = model(**batch)

			start_logits.append(outputs.start_logits.cpu().numpy())
			end_logits.append(outputs.end_logits.cpu().numpy())

		start_logits = np.concatenate(start_logits)
		end_logits = np.concatenate(end_logits)
		start_logits = start_logits[: len(features)]
		end_logits = end_logits[: len(features)]

		example_to_features = defaultdict(list)
		max_answer_length = 30
		n_best = 20
			
		for idx, feature in enumerate(features):
			example_to_features[feature["example_id"]].append(idx)

		for example in tqdm(examples, desc="Computing metrics"):
			# TODO: use squad in mrqa
			example_id = example["qid"]
			answers = []

			# Loop through all features associated with that example
			for feature_index in example_to_features[example_id]:
				start_logit = start_logits[feature_index]
				end_logit = end_logits[feature_index]
				offsets = features[feature_index]["offset_mapping"]

				start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
				end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
				for start_index in start_indexes:
					for end_index in end_indexes:
						# Skip answers that are not fully in the context
						if offsets[start_index] is None or offsets[end_index] is None:
							continue
						# Skip answers with a length that is either < 0 or > max_answer_length
						if (
							end_index < start_index
							or end_index - start_index + 1 > max_answer_length
						):
							continue

						answers.append(start_logit[start_index] + end_logit[end_index])

			
				if 1 < len(answers) < 200: # pad to same numbers of possible answers
					zero_list = [0] * (200 - len(answers))
					answers.extend(zero_list)
				elif len(answers) >= 200:
					answers = answers[:200]

				probs[i][feature_index] += torch.tensor(softmax(answers))

	return probs

def get_batch_prob_dropout_split(dataloader, device, features, examples, n_drop=10):
	## use tensor to save the answers

	c = 10

	if LOW_RES:
		model = AutoAdapterModel.from_pretrained(strategy_model_dir).to(device)
		model.train_adapter("unipelt") # activate adapter
	else:
		model = AutoModelForQuestionAnswering.from_pretrained(strategy_model_dir).to(device)
	
	model.train()

	probs = torch.zeros([n_drop, len(dataloader.dataset), c])
	
	for i in range(n_drop):
		start_logits = []
		end_logits = []
		for batch in tqdm(dataloader, desc="Evaluating_prob_dropout"):
			batch = {key: value.to(device) for key, value in batch.items()}
			with torch.no_grad():
				outputs = model(**batch)

			start_logits.append(outputs.start_logits.cpu().numpy())
			end_logits.append(outputs.end_logits.cpu().numpy())

		start_logits = np.concatenate(start_logits)
		end_logits = np.concatenate(end_logits)
		start_logits = start_logits[: len(features)]
		end_logits = end_logits[: len(features)]

		example_to_features = defaultdict(list)
		max_answer_length = 30
		n_best = 20
			
		for idx, feature in enumerate(features):
			example_to_features[feature["example_id"]].append(idx)

		for example in tqdm(examples, desc="Computing metrics"):
			# TODO: use squad in mrqa
			example_id = example["qid"]
			answers = []

			# Loop through all features associated with that example
			for feature_index in example_to_features[example_id]:
				start_logit = start_logits[feature_index]
				end_logit = end_logits[feature_index]
				offsets = features[feature_index]["offset_mapping"]

				start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
				end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
				for start_index in start_indexes:
					for end_index in end_indexes:
						# Skip answers that are not fully in the context
						if offsets[start_index] is None or offsets[end_index] is None:
							continue
						# Skip answers with a length that is either < 0 or > max_answer_length
						if (
							end_index < start_index
							or end_index - start_index + 1 > max_answer_length
						):
							continue

						answers.append(start_logit[start_index] + end_logit[end_index])

			
				if 1 < len(answers) < c: # pad to same numbers of possible answers
					zero_list = [0] * (c - len(answers))
					answers.extend(zero_list)
				elif len(answers) >= c:
					answers = answers[:c]

				probs[i][feature_index] += torch.tensor(softmax(answers))

	return probs

def get_embeddings(dataloader, device):
	if LOW_RES:
		model = AutoAdapterModel.from_pretrained(strategy_model_dir, output_hidden_states=True).to(device)
		model.train_adapter("unipelt") # activate adapter
	else:
		model = AutoModelForQuestionAnswering.from_pretrained(strategy_model_dir, output_hidden_states=True).to(device)
	
	model.eval()
	embeddings = torch.zeros([len(dataloader.dataset), model.config.to_dict()['hidden_size']])
	idxs_start = 0

	with torch.no_grad():
		for batch in tqdm(dataloader, desc="Evaluating_prob"):
			batch = {key: value.to(device) for key, value in batch.items()}
		
			outputs = model(**batch)

			hidden_states = outputs.hidden_states
			embedding_of_last_layer = hidden_states[-2][:, 0, :]

			idxs_end = idxs_start + len(hidden_states[-2])
			embeddings[idxs_start:idxs_end] = embedding_of_last_layer.cpu()
			idxs_start = idxs_end
		
	return embeddings

def get_grad_embeddings(dataloader, device, features, examples):
	if LOW_RES:
		model = AutoAdapterModel.from_pretrained(strategy_model_dir, output_hidden_states=True).to(device)
		model.train_adapter("unipelt") # activate adapter
	else:
		model = AutoModelForQuestionAnswering.from_pretrained(strategy_model_dir, output_hidden_states=True).to(device)
	
	model.eval()

	nLab = 20
	embDim = model.config.to_dict()['hidden_size']
	embeddings = np.zeros([len(dataloader.dataset), embDim * nLab])

	prob_dict = []
	idxs_start = 0

	with torch.no_grad():
		for batch in tqdm(dataloader, desc="Evaluating_prob"):
			batch = {key: Variable(value.to(device)) for key, value in batch.items()}
				
			# deepAL+: out, e1 = self.clf(x)
			outputs = model(**batch)
			# deepAL+: e1 = e1.data.cpu().numpy()
			hidden_states = outputs.hidden_states
			embedding_of_last_layer = hidden_states[-2][:, 0, :]
			embedding_of_last_layer = embedding_of_last_layer.data.cpu().numpy()

			# matually create features batch
			data_len_batch = len(outputs.start_logits)
			idxs_end = idxs_start + data_len_batch
			batch_idx = list(range(idxs_start, idxs_end))
			batch_feat = features.select(batch_idx)
			idxs_start = idxs_end

			out = logits_to_prob(outputs.start_logits.cpu().numpy(), outputs.end_logits.cpu().numpy(), batch_feat, batch_idx, examples, 200)
			batchProbs = F.softmax(out, dim=1).data.cpu().numpy()
			maxInds = np.argmax(batchProbs, 1)

			for j in range(data_len_batch):
				for c in range(nLab):
					if c == maxInds[j]:
						embeddings[batch_idx[j]][embDim * c : embDim * (c+1)] = deepcopy(embedding_of_last_layer[j]) * (1 - batchProbs[j][c]) * -1.0
					else:
						embeddings[batch_idx[j]][embDim * c : embDim * (c+1)] = deepcopy(embedding_of_last_layer[j]) * (-1 * batchProbs[j][c]) * -1.0
			
	return embeddings

def logits_to_prob(start_logits, end_logits, features, batch_idx, examples, num_classes):
	probs = torch.zeros([len(batch_idx), num_classes])
	
	example_to_features = defaultdict(list)
	max_answer_length = 30
	n_best = 20

	for idx, feature in enumerate(features):
		example_to_features[feature["example_id"]].append((idx, batch_idx[idx]))
	
	for example in examples:
		# TODO: use squad in mrqa
		example_id = example["qid"]
		answers = []
		
		# Loop through all features associated with that example
		for (feature_index, i) in example_to_features[example_id]:
			start_logit = start_logits[feature_index]
			end_logit = end_logits[feature_index]
			offsets = features[feature_index]["offset_mapping"]

			start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
			end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
			for start_index in start_indexes:
				for end_index in end_indexes:
					# Skip answers that are not fully in the context
					if offsets[start_index] is None or offsets[end_index] is None:
						continue
					# Skip answers with a length that is either < 0 or > max_answer_length
					if (
						end_index < start_index
						or end_index - start_index + 1 > max_answer_length
					):
						continue
					answers.append(start_logit[start_index] + end_logit[end_index])


			if 1 < len(answers) < num_classes: # pad to same numbers of possible answers
				zero_list = [0] * (num_classes - len(answers))
				answers.extend(zero_list)
			elif len(answers) >= num_classes:
				answers = answers[:num_classes]
			probs[feature_index] = torch.tensor(answers)

	return probs

def softmax(x):
	"""Compute softmax values for each sets of scores in x."""
	return np.exp(x) / np.sum(np.exp(x), axis=0)