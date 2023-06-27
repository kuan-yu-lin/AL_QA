from torch.utils.data import DataLoader
from transformers import default_data_collator

import numpy as np
import torch
import collections
from tqdm.auto import tqdm

from utils import get_unlabel_data
from model import get_prob, get_prob_dropout, get_prob_dropout_split

def random_sampling_query(labeled_idxs, n):
    return np.random.choice(np.where(labeled_idxs==0)[0], n, replace=False)

def margin_sampling_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(
		unlabeled_data,
		shuffle=True,
		collate_fn=default_data_collator,
		batch_size=8,
	)
    # TODO: print for recording
    print('Margin querying starts!')
    prob_dict = get_prob(unlabeled_dataloader, device, unlabeled_features, examples)
    # TODO: print for recording
    print('Got probability!')
    uncertainties_dict = {}
    for idx, probs in prob_dict.items():
        if len(probs) > 1: # if prob_dict['probs'] is not 0
            sort_probs = np.sort(probs)[::-1] # This method returns a copy of the array, leaving the original array unchanged.
            uncertainties_dict[idx] = sort_probs[0] - sort_probs[1]
        elif idx:
            uncertainties_dict[idx] = np.array([0])

    sorted_uncertainties_list = sorted(uncertainties_dict.items(), key=lambda x: x[1], reverse=True)
    return unlabeled_idxs[[idx for (idx, uncertainties) in sorted_uncertainties_list[:n]]]

def least_confidence_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(
		unlabeled_data,
		shuffle=True,
		collate_fn=default_data_collator,
		batch_size=8,
	)
    # TODO: print for recording
    print('LC querying starts!')
    prob_dict = get_prob(unlabeled_dataloader, device, unlabeled_features, examples)
    # TODO: print for recording
    print('Got probability!')

    confidence_dict = {}
    for idx, probs in prob_dict.items():
        if len(probs) > 1: # if prob_dict['probs'] is not 0
            confidence_dict[idx] = max(probs)
        elif idx:
            confidence_dict[idx] = np.array([0])

    sorted_confidence_list = sorted(confidence_dict.items(), key=lambda x: x[1])
    return unlabeled_idxs[[idx for (idx, confidence) in sorted_confidence_list[:n]]]

def var_ratio_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(
		unlabeled_data,
		shuffle=True,
		collate_fn=default_data_collator,
		batch_size=8,
	)
    # TODO: print for recording
    print('Var Ratio querying starts!')
    prob_dict = get_prob(unlabeled_dataloader, device, unlabeled_features, examples)
    # TODO: print for recording
    print('Got probability!')
    confidence_dict = {}
    for idx, probs in prob_dict.items():
        if len(probs) > 1: # if prob_dict['probs'] is not 0
            confidence_dict[idx] = 1.0 - max(probs)
        elif idx:
            confidence_dict[idx] = np.array([0])

    sorted_confidence_list = sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)
    return unlabeled_idxs[[idx for (idx, confidence) in sorted_confidence_list[:n]]]

def entropy_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(
		unlabeled_data,
		shuffle=True,
		collate_fn=default_data_collator,
		batch_size=8,
	)
    # TODO: print for recording
    print('Entropy querying starts!')
    prob_dict = get_prob(unlabeled_dataloader, device, unlabeled_features, examples)
    # TODO: print for recording
    print('Got probability!')
    entropy_dict = {}
    for idx, probs in prob_dict.items():
        if len(probs) > 1: # if prob_dict['probs'] is not 0
            log_probs = np.log(probs)
            entropy_dict[idx] = (probs*log_probs).sum()
        elif idx:
            entropy_dict[idx] = np.array([0])
    sorted_entropy_list = sorted(entropy_dict.items(), key=lambda x: x[1])
    return unlabeled_idxs[[idx for (idx, entropy) in sorted_entropy_list[:n]]]

def margin_sampling_dropout_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(
		unlabeled_data,
		shuffle=True,
		collate_fn=default_data_collator,
		batch_size=8,
	)
    # TODO: print for recording
    print('Margin dropout querying starts!')
    prob_dict = get_prob_dropout(unlabeled_dataloader, device, unlabeled_features, examples)
    # TODO: print for recording
    print('Got probability!')
    uncertainties_dict = {}
    for idx, probs in prob_dict.items():
        if len(probs) > 1: # if prob_dict['probs'] is not 0
            sort_probs = np.sort(probs)[::-1] # This method returns a copy of the array, leaving the original array unchanged.
            uncertainties_dict[idx] = sort_probs[0] - sort_probs[1]
        elif idx:
            uncertainties_dict[idx] = np.array([0])

    sorted_uncertainties_list = sorted(uncertainties_dict.items(), key=lambda x: x[1], reverse=True)
    return unlabeled_idxs[[idx for (idx, uncertainties) in sorted_uncertainties_list[:n]]]

def least_confidence_dropout_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(
		unlabeled_data,
		shuffle=True,
		collate_fn=default_data_collator,
		batch_size=8,
	)
    # TODO: print for recording
    print('LC dropout querying starts!')
    prob_dict = get_prob_dropout(unlabeled_dataloader, device, unlabeled_features, examples)
    # TODO: print for recording
    print('Got probability!')

    confidence_dict = {}
    for idx, probs in prob_dict.items():
        if len(probs) > 1: # if prob_dict['probs'] is not 0
            confidence_dict[idx] = max(probs)
        elif idx:
            confidence_dict[idx] = np.array([0])

    sorted_confidence_list = sorted(confidence_dict.items(), key=lambda x: x[1])
    return unlabeled_idxs[[idx for (idx, confidence) in sorted_confidence_list[:n]]]

def entropy_dropout_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(
		unlabeled_data,
		shuffle=True,
		collate_fn=default_data_collator,
		batch_size=8,
	)
    # TODO: print for recording
    print('Entropy dropout querying starts!')
    prob_dict = get_prob_dropout(unlabeled_dataloader, device, unlabeled_features, examples)
    # TODO: print for recording
    print('Got probability!')
    entropy_dict = {}
    for idx, probs in prob_dict.items():
        if len(probs) > 1: # if prob_dict['probs'] is not 0
            log_probs = np.log(probs)
            entropy_dict[idx] = (probs*log_probs).sum()
        elif idx:
            entropy_dict[idx] = np.array([0])
    sorted_entropy_list = sorted(entropy_dict.items(), key=lambda x: x[1])
    return unlabeled_idxs[[idx for (idx, entropy) in sorted_entropy_list[:n]]]

def bayesian_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(
      unlabeled_data,
      shuffle=True,
      collate_fn=default_data_collator,
      batch_size=8,
    )
    # TODO: print for recording
    print('BALD querying starts!')
    probs = get_prob_dropout_split(unlabeled_dataloader, device, unlabeled_features, examples)
    # TODO: print for recording
    print('Got probability!')
    probs_mean = probs.mean(0)
    entropy1 = (-probs_mean*torch.log(probs_mean)).sum(1)
    entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
    uncertainties = entropy2 - entropy1
    # later on, we can use batch
    return unlabeled_idxs[uncertainties.sort()[1][:n]]

def mean_std_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(
  		unlabeled_data,
      shuffle=True,
      collate_fn=default_data_collator,
      batch_size=8,
    )
    # TODO: print for recording
    print('Mean STD querying starts!')
    probs = get_prob_dropout_split(unlabeled_dataloader, device, unlabeled_features, examples).numpy()
    # TODO: print for recording
    print('Got probability!')
    sigma_c = np.std(probs, axis=0)
    uncertainties = torch.from_numpy(np.mean(sigma_c, axis=-1)) # use tensor.sort() will sort the data and produce sorted indexes
    return unlabeled_idxs[uncertainties.sort(descending=True)[1][:n]]

def kmeans_query():
    pass

def kcenter_query():
    pass

def badge_query():
    pass

def loss_prediction_query():
    pass

def ceal_query():
    pass
