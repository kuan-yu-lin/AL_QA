from torch.utils.data import DataLoader
from transformers import default_data_collator

import sys
sys.path.insert(0, './')
from strategies.sub_utils import get_unlabel_data, init_centers, get_us, get_us_uc, sub_decode_id
from strategies.sub_model import get_grad_embeddings
import arguments

args_input = arguments.get_args()
NUM_QUERY = args_input.batch
MODEL_BATCH = args_input.model_batch
# UNIQ_CONTEXT = args_input.unique_context
_, _, _, _, UNIQ_CONTEXT = sub_decode_id()
if UNIQ_CONTEXT: n = NUM_QUERY*10
else: n = NUM_QUERY*2

def badge(n_pool, labeled_idxs, dataset, features, examples, device, i):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, dataset)
    unlabeled_features = features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(unlabeled_data,
                                      collate_fn=default_data_collator,
                                      batch_size=MODEL_BATCH,
                                    )
    print('BADGE querying starts.')
    print('Query {} data from {} unlabeled training data.\n'.format(NUM_QUERY, len(unlabeled_data)))

    gradEmbedding = get_grad_embeddings(unlabeled_dataloader, device, unlabeled_features, examples)
    print('Got embeddings.')
    chosen = init_centers(gradEmbedding, n)
    score_ordered_idxs = unlabeled_idxs[chosen]
    
    if UNIQ_CONTEXT:
        iter_i_labeled_idxs = get_us_uc(labeled_idxs, score_ordered_idxs, n_pool, features, i)
    else:
        iter_i_labeled_idxs = get_us(labeled_idxs, score_ordered_idxs, n_pool, features, i)

    return iter_i_labeled_idxs