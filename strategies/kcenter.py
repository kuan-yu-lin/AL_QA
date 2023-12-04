from torch.utils.data import DataLoader
from transformers import default_data_collator
from tqdm.auto import tqdm
import numpy as np
from sklearn.decomposition import PCA
import sys
sys.path.insert(0, './')

from strategies.sub_utils import get_us, get_us_uc, get_us_ue
from strategies.sub_model import get_embeddings
import arguments

args_input = arguments.get_args()
NUM_QUERY = args_input.batch
MODEL_BATCH = args_input.model_batch
UNIQ_CONTEXT = args_input.uni_con
DIST_EMBED = args_input.dist_embed
if UNIQ_CONTEXT: n = NUM_QUERY*10
elif DIST_EMBED: n = NUM_QUERY*10
else: n = NUM_QUERY*5

def kcenter(n_pool, labeled_idxs, dataset, features, device, i):
	labeled_idxs_in_query = labeled_idxs.copy()
	train_dataloader = DataLoader(dataset,
								  collate_fn=default_data_collator,
								  batch_size=MODEL_BATCH,
								)
	print('KCenter greedy querying starts.')
	print('Query {} data.'.format(NUM_QUERY))
	
	embeddings = get_embeddings(train_dataloader, device)
	print('Got embeddings.')
	embeddings = embeddings.numpy()

	dist_mat = np.matmul(embeddings, embeddings.transpose())
	sq = np.array(dist_mat.diagonal()).reshape(len(labeled_idxs_in_query), 1)
	dist_mat *= -2
	dist_mat += sq
	dist_mat += sq.transpose()
	dist_mat = np.sqrt(dist_mat)

	mat = dist_mat[~labeled_idxs_in_query, :][:, labeled_idxs_in_query]

	for ii in tqdm(range(n), ncols=100):
		mat_min = mat.min(axis=1)
		q_idx_ = mat_min.argmax()
		q_idx = np.arange(n_pool)[~labeled_idxs_in_query][q_idx_]
		labeled_idxs_in_query[q_idx] = True
		mat = np.delete(mat, q_idx_, 0)
		mat = np.append(mat, dist_mat[~labeled_idxs_in_query, q_idx][:, None], axis=1)
	
	score_ordered_idxs = np.arange(n_pool)[(labeled_idxs ^ labeled_idxs_in_query)]
	
	if UNIQ_CONTEXT:
		iter_i_labeled_idxs, ssi_ = get_us_uc(labeled_idxs, score_ordered_idxs, n_pool, features, i)
	elif DIST_EMBED:
		iter_i_labeled_idxs, ssi_ = get_us_ue(labeled_idxs, score_ordered_idxs, n_pool, dataset, features, device, i)
	else:
		iter_i_labeled_idxs, ssi_ = get_us(labeled_idxs, score_ordered_idxs, n_pool, features, i)

	return iter_i_labeled_idxs, ssi_
