from torch.utils.data import DataLoader, random_split
from transformers import default_data_collator
import torch
import numpy as np
import sys
sys.path.insert(0, './')

from strategies.sub_utils import get_unlabel_data, H, class_combinations, get_us, get_us_uc, sub_decode_id
from strategies.sub_model import get_batch_prob_dropout_split
import arguments

args_input = arguments.get_args()
MODEL_BATCH = args_input.model_batch
NUM_QUERY = args_input.batch
# LOW_RES = args_input.low_resource
# UNIQ_CONTEXT = args_input.unique_context
LOW_RES, _, _, _, UNIQ_CONTEXT = sub_decode_id()
if UNIQ_CONTEXT: n = NUM_QUERY*10
else: n = NUM_QUERY*5

def batch_bald(n_pool, labeled_idxs, dataset, features, examples, device, i):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, dataset)
    print('BatchBALD querying starts.')
    print('Query {} data from {} unlabeled training data.\n'.format(NUM_QUERY, len(unlabeled_data)))
    m = 1e4  # number of MC samples for label combinations
    if LOW_RES:
        num_sub_pool = len(unlabeled_data)
    else:
        num_sub_pool = 2000  # number of datapoints in the subpool from which we acquire

    c = 10
    k = 10

    # performing BatchBALD on the whole pool is very expensive, so we take a random subset of the pool.
    num_extra = len(unlabeled_data) - num_sub_pool

    if num_extra > 0:
        sub_pool_data, _ = random_split(unlabeled_data, [num_sub_pool, num_extra])
        sub_pool_idxs = np.array(sub_pool_data.indices)
    else:
        # even if we don't have enough data left to split, we still need to
        # call random_splot to avoid messing up the indexing later on
        sub_pool_data, _ = random_split(unlabeled_data, [len(unlabeled_data), 0])
        sub_pool_idxs = np.array(sub_pool_data.indices)

    # forward pass on the pool once to get class probabilities for each x
    pool_loader = DataLoader(sub_pool_data,
        batch_size=MODEL_BATCH, pin_memory=True, shuffle=False, collate_fn=default_data_collator)
    pool_features = features.select(sub_pool_idxs)

    pool_p_y = get_batch_prob_dropout_split(pool_loader, device, pool_features, examples, n_drop=k).permute(1, 0, 2) 
    # pool_p_y.shape = n * k * c = 500 * 10 * 10
    # this only need to be calculated once so we pull it out of the loop
    H2 = (H(pool_p_y).sum(axis=(1,2))/k)

    # get all class combinations
    c_1_to_n = class_combinations(c, n, m) # tensor of size [m * n]

    # tensor of size [m * k]
    p_y_1_to_n_minus_1 = None

    # store the indices of the chosen datapoints in the subpool
    best_sub_local_indices = []

    # create a mask to keep track of which indices we've chosen
    remaining_indices = torch.ones(len(sub_pool_data), dtype=bool)

    for batch_n in range(n):
        # tensor of size [N * m * l] # [500, 10000, 3]
        p_y_n = pool_p_y[:, c_1_to_n[:, batch_n], :] # k should be >= c 

        # tensor of size [N * m * k]   
        if p_y_1_to_n_minus_1 == None:
            p_y_1_to_n = p_y_n
        elif torch.tensor(0) in p_y_1_to_n_minus_1:
            p_y_1_to_n = p_y_n
        else:
            p_y_1_to_n = torch.einsum('mk,pmk->pmk', p_y_1_to_n_minus_1, p_y_n)

        # and compute the left entropy term
        H1 = H(p_y_1_to_n.mean(axis=2)).sum(axis=1)

        # scores is a vector of scores for each element in the pool.
        # mask by the remaining indices and find the highest scoring element
        scores = H1 - H2
        
        best_local_index = torch.argmax(scores - max(scores)*(~remaining_indices)).item()

        best_sub_local_indices.append(best_local_index)
        # save the computation for the next batch
        p_y_1_to_n_minus_1 = p_y_1_to_n[best_local_index]
        # remove the chosen element from the remaining indices mask
        remaining_indices[best_local_index] = False
        
    # we've subset-ed our dataset twice, so we need to go back through
    # subset indices twice to recover the global indices of the chosen data
    best_local_indices = np.array(sub_pool_idxs)[best_sub_local_indices]
    best_global_indices = np.array(unlabeled_idxs)[best_local_indices]
    print('We got {} best global indices.'.format(len(best_global_indices)))

    if UNIQ_CONTEXT:
        iter_i_labeled_idxs = get_us_uc(labeled_idxs, best_global_indices, n_pool, features, i)
    else:
        iter_i_labeled_idxs = get_us(labeled_idxs, best_global_indices, n_pool, features, i)
    
    return iter_i_labeled_idxs
