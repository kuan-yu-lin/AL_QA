import numpy as np
import collections
from tqdm.auto import tqdm

from utils import get_unlabel_data
from evaluations import get_prob, get_prob_dropout

def random_sampling_query(labeled_idxs, n):
    return np.random.choice(np.where(labeled_idxs==0)[0], n, replace=False)

def margin_sampling_query(n_pool, labeled_idxs, train_dataset, trainer_qs, example, n):
    
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    prob_list_dict = get_prob(trainer_qs, unlabeled_data, example)

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
    
    sorted_uncertainties_dict = sorted(uncertainties_list_dict, key=lambda d: d['uncertainties'])   
    return unlabeled_idxs[[uncertainties_dict['idx'][0] for uncertainties_dict in sorted_uncertainties_dict[:n]]]

def least_confidence_query(n_pool, labeled_idxs, train_dataset, trainer_qs, example, n):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    prob_list_dict = get_prob(trainer_qs, unlabeled_data, example)

    confidence_list_dict = []
    for d in prob_list_dict:
        if len(d['probs']) > 1: # if prob_dict['probs'] is not 0
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
    return unlabeled_idxs[[confidence_dict['idx'][0] for confidence_dict in sorted_confidence_dict[:n]]]

def var_ratio_query(n_pool, labeled_idxs, train_dataset, trainer_qs, example, n):
    unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    probs_list_dict = get_prob(trainer_qs, unlabeled_data, example)

    confidence_list_dict = []
    for d in probs_list_dict:
        if len(d['probs']) > 1: # if prob_dict['probs'] is not 0
            confidence = max(d['probs'])
            confidence_list_dict.append(
                {'idx': d['idx'], 
                    'confidence': 1 - confidence}
                    )
        elif d['idx']:
            confidence_list_dict.append(
                {'idx': d['idx'], 
                    'confidence': np.array([0])}
                    )

    sorted_confidence_dict = sorted(confidence_list_dict, key=lambda d: d['confidence'], reverse=True)
    return [confidence_dict['idx'][0] for confidence_dict in sorted_confidence_dict[:n]]

def entropy_query(n_pool, labeled_idxs, train_dataset, trainer_qs, example, n):

    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    prob_list_dict = get_prob(trainer_qs, unlabeled_data, example)

    entropy_list_dict = []
    for d in prob_list_dict:
        if len(d['probs']) > 1: 
            log_probs = np.log(d['probs'])
            entropy_list_dict.append(
                {'idx': d['idx'], 
                'entropy': (d['probs']*log_probs).sum()}
                )
        elif d['idx']:
            entropy_list_dict.append(
                {'idx': d['idx'], 
                'entropy': np.array([0])}
                )

    sorted_entropy_dict = sorted(entropy_list_dict, key=lambda d: d['entropy'], reverse=True) # use largest Entropy, different from deepAL+ code
    return unlabeled_idxs[[entropy_dict['idx'][0] for entropy_dict in sorted_entropy_dict[:n]]]

def margin_sampling_dropout_query(n_pool, labeled_idxs, train_dataset, trainer_qs, example, n):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    prob_list_dict = get_prob_dropout(trainer_qs, unlabeled_data, example)

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
    
    sorted_uncertainties_dict = sorted(uncertainties_list_dict, key=lambda d: d['uncertainties'])   
    return unlabeled_idxs[[uncertainties_dict['idx'][0] for uncertainties_dict in sorted_uncertainties_dict[:n]]]

def least_confidence_dropout_query(n_pool, labeled_idxs, train_dataset, trainer_qs, example, n):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    prob_list_dict = get_prob_dropout(trainer_qs, unlabeled_data, example)

    confidence_list_dict = []
    for d in prob_list_dict:
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
    return unlabeled_idxs[[confidence_dict['idx'][0] for confidence_dict in sorted_confidence_dict[:n]]]

def entropy_dropout_query(n_pool, labeled_idxs, train_dataset, trainer_qs, example, n):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    prob_list_dict = get_prob_dropout(trainer_qs, unlabeled_data, example)
    entropy_list_dict = []
    for d in prob_list_dict:
        if len(d['probs']) > 1: # if prob_dict['probs'] is not 0
            log_probs = np.log(d['probs'])
            entropy_list_dict.append(
                {'idx': d['idx'], 
                'entropy': (d['probs']*log_probs).sum()}
                )
        elif d['idx']:
            entropy_list_dict.append(
                {'idx': d['idx'], 
                'entropy': np.array([0])}
                )
    # deepAL+: return unlabeled_idxs[uncertainties.sort()[1][:n]] # use smallest Entropy
    sorted_entropy_dict = sorted(entropy_list_dict, key=lambda d: d['entropy'], reverse=True) # use largest Entropy, different from deepAL+ code
    return unlabeled_idxs[[entropy_dict['idx'][0] for entropy_dict in sorted_entropy_dict[:n]]]

def bayesian_query():
    pass

def mean_std_query():
    pass

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
