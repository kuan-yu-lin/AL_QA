import numpy as np

import sys
sys.path.insert(0, './')
from strategies.sub_utils import get_unlabel_data, get_us, get_us_uc, get_us_ue
import arguments

args_input = arguments.get_args()
NUM_QUERY = args_input.batch
UNIQ_CONTEXT = args_input.uni_con
DIST_EMBED = args_input.dist_embed

def random_sampling(n_pool, labeled_idxs, dataset, features, device, i):
	print('Random querying starts.')
	unlabeled_idxs, _ = get_unlabel_data(n_pool, labeled_idxs, dataset)
	score_ordered_idxs = np.random.choice(np.where(labeled_idxs==0)[0], len(unlabeled_idxs), replace=False)
	if UNIQ_CONTEXT:
		iter_i_labeled_idxs, ssi_ = get_us_uc(labeled_idxs, score_ordered_idxs, n_pool, features, i)
	elif DIST_EMBED:
		iter_i_labeled_idxs, ssi_ = get_us_ue(labeled_idxs, score_ordered_idxs, n_pool, dataset, features, device, i)
	else:
		iter_i_labeled_idxs, ssi_ = get_us(labeled_idxs, score_ordered_idxs, n_pool, features, i)

	return iter_i_labeled_idxs, ssi_