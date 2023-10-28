from torch.utils.data import DataLoader
from transformers import default_data_collator
import numpy as np
import sys
sys.path.insert(0, './')

from utils import get_unlabel_data
from model import get_prob_dropout
import arguments

args_input = arguments.get_args()
MODEL_BATCH = args_input.model_batch 

def least_confidence_dropout(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(
		unlabeled_data,
		collate_fn=default_data_collator,
		batch_size=MODEL_BATCH,
	)
    
    print('LC dropout querying starts.')
    print('Query {} data from {} unlabeled training data.\n'.format(n, len(unlabeled_data)))
    
    prob_dict = get_prob_dropout(unlabeled_dataloader, device, unlabeled_features, examples, n_drop=10)
    print('Got probability.')

    confidence_dict = {}
    for idx, probs in prob_dict.items():
        if len(probs) > 1: # if prob_dict['probs'] is not 0
            confidence_dict[idx] = max(probs)
        elif idx:
            confidence_dict[idx] = np.array([0])

    sorted_confidence_list = sorted(confidence_dict.items(), key=lambda x: x[1])
    return unlabeled_idxs[[idx for (idx, confidence) in sorted_confidence_list[:n]]]