import numpy as np
import collections
from tqdm.auto import tqdm

import arguments
from utils import get_preds, get_unlabel_data
from evaluations import get_prob

args_input = arguments.get_args()
NUM_QUERY = args_input.batch

def random_sampling_query(labeled_idxs):
    return np.random.choice(np.where(labeled_idxs==0)[0], NUM_QUERY, replace=False)

def margin_sampling_query(n_pool, labeled_idxs, train_dataset, trainer_qs, example):
    
    unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    start_logits, end_logits, = get_preds(trainer_qs, unlabeled_data)
    prob_list_dict = get_prob(start_logits, end_logits, unlabeled_data, example)

    uncertainties_list_dict = []
    for d in prob_list_dict:
        if len(d['probs']) > 1: # if prob_dict['probs'] is not 0
            uncertainties = d['probs'][0] - d['probs'][1]
            uncertainties_list_dict.append(
                {'idx': d['idx'], 
                 'uncertainties': uncertainties}
                 )
        elif d['idx']:
            uncertainties_list_dict.append(
                {'idx': d['idx'], 
                 'uncertainties': np.array([0])}
                 )
    
    sorted_uncertainties_dict = sorted(uncertainties_list_dict, key=lambda d: d['uncertainties']) # TODO: check smallest or largest, now is smallest    
    return [uncertainties_dict['idx'][0] for uncertainties_dict in sorted_uncertainties_dict[:NUM_QUERY]]

def least_confidence_query(n_pool, labeled_idxs, train_dataset, trainer_qs, example):

    unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    start_logits, end_logits, = get_preds(trainer_qs, unlabeled_data)
    probs_list_dict = get_prob(start_logits, end_logits, unlabeled_data, example)

    confidence_list_dict = []
    for d in probs_list_dict:
        if len(d['probs']) > 1: # if prob_dict['probs'] is not 0
            # uncertainties_d = d['probs']
            confidence = max(d['probs'])
            confidence_list_dict.append(
                {'idx': d['idx'], 
                    'confidence': confidence}
                    )
        elif d['idx']:
            confidence_list_dict.append(
                {'idx': d['idx'], 
                    'confidence': np.array([0])}
                    )

    sorted_confidence_dict = sorted(confidence_list_dict, key=lambda d: d['confidence'])
    return [confidence_dict['idx'][0] for confidence_dict in sorted_confidence_dict[:NUM_QUERY]]

def var_ratio_query(start_logits, end_logits, features, examples):
    pass

def entropy_query():
    # probs = self.predict_prob(unlabeled_data) # I always get score instead
    # log_probs = torch.log(probs)
    # uncertainties = (probs*log_probs).sum(1)
    # return unlabeled_idxs[uncertainties.sort()[1][:n]] # same as other query, the n smallest
    pass

def margin_sampling_dropout_query():
    # print to see the size difference, what was dropped
    # probs /= n_drop
    pass

def least_confidence_dropout_query():
    pass

def entropy_dropout_query():
    pass

def kmeans_query():
    pass

def kcenter_query():
    pass

def bayesian_query():
    pass

def mean_std_query():
    pass

def badge_query():
    pass

def loss_prediction_query():
    pass

def ceal_query():
    pass
