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

def margin_dropout(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(
		unlabeled_data,
		collate_fn=default_data_collator,
		batch_size=MODEL_BATCH,
	)
    
    print('Margin dropout querying starts.')
    print('Query {} data from {} unlabeled training data.\n'.format(n, len(unlabeled_data)))

    prob_dict = get_prob_dropout(unlabeled_dataloader, device, unlabeled_features, examples, n_drop=10)
    print('Got probability.')
    uncertainties_dict = {}
    for idx, probs in prob_dict.items():
        if len(probs) > 1: # if prob_dict['probs'] is not 0
            sort_probs = np.sort(probs)[::-1] # This method returns a copy of the array, leaving the original array unchanged.
            uncertainties_dict[idx] = sort_probs[0] - sort_probs[1]
        elif idx:
            uncertainties_dict[idx] = np.array([0])

    sorted_uncertainties_list = sorted(uncertainties_dict.items(), key=lambda x: x[1], reverse=True)
    return unlabeled_idxs[[idx for (idx, uncertainties) in sorted_uncertainties_list[:n]]]