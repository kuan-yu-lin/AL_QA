from torch.utils.data import DataLoader
from transformers import default_data_collator
import numpy as np
import sys
sys.path.insert(0, './')

from strategies.sub_utils import get_unlabel_data, get_us, get_us_uc, sub_decode_id
from strategies.sub_model import get_prob_dropout
import arguments

args_input = arguments.get_args()
NUM_QUERY = args_input.batch
MODEL_BATCH = args_input.model_batch
UNIQ_CONTEXT = args_input.uni_con

def least_confidence_dropout(n_pool, labeled_idxs, dataset, features, examples, device, i):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, dataset)
    unlabeled_features = features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(
		unlabeled_data,
		collate_fn=default_data_collator,
		batch_size=MODEL_BATCH,
	)
    
    print('LC dropout querying starts.')
    print('Query {} data from {} unlabeled training data.\n'.format(NUM_QUERY, len(unlabeled_data)))
    
    prob_dict = get_prob_dropout(unlabeled_dataloader, device, unlabeled_features, examples, n_drop=10)
    print('Got probability.')

    confidence_dict = {}
    for idx, probs in prob_dict.items():
        if len(probs) > 1: # if prob_dict['probs'] is not 0
            confidence_dict[idx] = max(probs)
        elif idx:
            confidence_dict[idx] = np.array([0])

    sorted_confidence_list = sorted(confidence_dict.items(), key=lambda x: x[1])
    score_ordered_idxs = unlabeled_idxs[[idx for (idx, _) in sorted_confidence_list]]
    
    if UNIQ_CONTEXT:
        iter_i_labeled_idxs = get_us_uc(labeled_idxs, score_ordered_idxs, n_pool, features, i)
    else:
        iter_i_labeled_idxs = get_us(labeled_idxs, score_ordered_idxs, n_pool, features, i)

    return iter_i_labeled_idxs