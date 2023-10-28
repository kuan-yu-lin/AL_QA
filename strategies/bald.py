from torch.utils.data import DataLoader
from transformers import default_data_collator
import torch
import sys
sys.path.insert(0, './')

from utils import get_unlabel_data, H
from model import get_prob_dropout_split
import arguments

args_input = arguments.get_args()
MODEL_BATCH = args_input.model_batch

def bald(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(
        unlabeled_data,
        collate_fn=default_data_collator,
        batch_size=MODEL_BATCH,
    )
    print('BALD querying starts.')
    print('Query {} data from {} unlabeled training data.\n'.format(n, len(unlabeled_data)))
    
    probs = get_prob_dropout_split(unlabeled_dataloader, device, unlabeled_features, examples, n_drop=10)
    print('Got probability.')
    probs_mean = probs.mean(0)
    entropy1 = (-probs_mean*torch.log(probs_mean)).sum(1)
    entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
    uncertainties = entropy2 - entropy1
    # I(y;W | x) = H1 - H2 = H(y|x) - E_w[H(y|x,W)]
    return unlabeled_idxs[uncertainties.sort()[1][:n]]
