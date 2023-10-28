from torch.utils.data import DataLoader
from transformers import default_data_collator
import numpy as np
import sys
sys.path.insert(0, './')

from utils import get_unlabel_data
from model import get_prob
import arguments

args_input = arguments.get_args()
MODEL_BATCH = args_input.model_batch

def entropy(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(
		unlabeled_data,
		collate_fn=default_data_collator,
		batch_size=MODEL_BATCH,
	)
    
    print('Entropy querying starts.')
    print('Query {} data from {} unlabeled training data.\n'.format(n, len(unlabeled_data)))

    prob_dict = get_prob(unlabeled_dataloader, device, unlabeled_features, examples)
    print('Got probability.')
    entropy_dict = {}
    for idx, probs in prob_dict.items():
        if len(probs) > 1: # if prob_dict['probs'] is not 0
            log_probs = np.log(probs)
            entropy_dict[idx] = (probs*log_probs).sum()
        elif idx:
            entropy_dict[idx] = np.array([0])
    sorted_entropy_list = sorted(entropy_dict.items(), key=lambda x: x[1])
    return unlabeled_idxs[[idx for (idx, entropy) in sorted_entropy_list[:n]]]