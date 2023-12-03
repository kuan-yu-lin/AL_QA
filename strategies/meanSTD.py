from torch.utils.data import DataLoader
from transformers import default_data_collator
import torch
import numpy as np
import sys
sys.path.insert(0, './')

from strategies.sub_utils import get_unlabel_data, get_us, get_us_uc, get_us_ue
from strategies.sub_model import get_prob_dropout_split
import arguments

args_input = arguments.get_args()
NUM_QUERY = args_input.batch
MODEL_BATCH = args_input.model_batch
UNIQ_CONTEXT = args_input.uni_con
DIST_EMBED = args_input.dist_embed

def mean_std(n_pool, labeled_idxs, dataset, features, examples, device, i):
	unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, dataset)
	unlabeled_features = features.select(unlabeled_idxs)
	unlabeled_dataloader = DataLoader(
  		unlabeled_data,
		collate_fn=default_data_collator,
		batch_size=MODEL_BATCH,
	)
	print('Mean STD querying starts.')
	print('Query {} data from {} unlabeled training data.\n'.format(NUM_QUERY, len(unlabeled_data)))
	
	probs = get_prob_dropout_split(unlabeled_dataloader, device, unlabeled_features, examples, n_drop=10).numpy()
	print('Got probability.')
	sigma_c = np.std(probs, axis=0)
	uncertainties = torch.from_numpy(np.mean(sigma_c, axis=-1)) # use tensor.sort() will sort the data and produce sorted indexes

	score_ordered_idxs = unlabeled_idxs[uncertainties.sort(descending=True)[1]]
	
	if UNIQ_CONTEXT:
		iter_i_labeled_idxs, ssi_ = get_us_uc(labeled_idxs, score_ordered_idxs, n_pool, features, i)
	elif DIST_EMBED:
		iter_i_labeled_idxs, ssi_ = get_us_ue(labeled_idxs, score_ordered_idxs, n_pool, dataset, features, device, i)
	else:
		iter_i_labeled_idxs, ssi_ = get_us(labeled_idxs, score_ordered_idxs, n_pool, features, i)

	return iter_i_labeled_idxs