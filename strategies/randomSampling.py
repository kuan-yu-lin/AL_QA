import numpy as np

import sys
sys.path.insert(0, './')
from strategies.sub_utils import get_us, get_us_uc, sub_decode_id
import arguments

args_input = arguments.get_args()
NUM_QUERY = args_input.batch
# LOW_RES = args_input.low_resource
# UNIQ_CONTEXT = args_input.unique_context
LOW_RES, _, _, _, UNIQ_CONTEXT = sub_decode_id()
if UNIQ_CONTEXT: n = NUM_QUERY*10
else: n = NUM_QUERY*5

def random_sampling(n_pool, labeled_idxs, features, i):
    print('Random querying starts.')

    score_ordered_idxs = np.random.choice(np.where(labeled_idxs==0)[0], n, replace=False)
    if UNIQ_CONTEXT:
        iter_i_labeled_idxs = get_us_uc(labeled_idxs, score_ordered_idxs, n_pool, features, i)
    else:
        iter_i_labeled_idxs = get_us(labeled_idxs, score_ordered_idxs, n_pool, features, i)

    return iter_i_labeled_idxs