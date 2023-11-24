from torch.utils.data import DataLoader
from transformers import default_data_collator
import torch
import sys
sys.path.insert(0, './')

from strategies.sub_utils import get_unlabel_data, H, get_us, get_us_uc
from strategies.sub_model import get_prob_dropout_split
import arguments

args_input = arguments.get_args()
NUM_QUERY = args_input.batch
MODEL_BATCH = args_input.model_batch
UNIQ_CONTEXT = args_input.uni_con

def bald(n_pool, labeled_idxs, dataset, features, examples, device, i):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, dataset)
    unlabeled_features = features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(
        unlabeled_data,
        collate_fn=default_data_collator,
        batch_size=MODEL_BATCH,
    )
    print('BALD querying starts.')
    print('Query {} data from {} unlabeled training data.\n'.format(NUM_QUERY, len(unlabeled_data)))
    
    probs = get_prob_dropout_split(unlabeled_dataloader, device, unlabeled_features, examples, n_drop=10)
    print('Got probability.')
    probs_mean = probs.mean(0)
    entropy1 = (-probs_mean*torch.log(probs_mean)).sum(1)
    entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
    uncertainties = entropy2 - entropy1
    # I(y;W | x) = H1 - H2 = H(y|x) - E_w[H(y|x,W)]
    score_ordered_idxs = unlabeled_idxs[uncertainties.sort()[1]]

    if UNIQ_CONTEXT:
        iter_i_labeled_idxs = get_us_uc(labeled_idxs, score_ordered_idxs, n_pool, features, i)
    else:
        iter_i_labeled_idxs = get_us(labeled_idxs, score_ordered_idxs, n_pool, features, i)

    return iter_i_labeled_idxs