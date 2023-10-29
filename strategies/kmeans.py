from torch.utils.data import DataLoader
from transformers import default_data_collator
from sklearn.cluster import KMeans
import numpy as np
import sys
sys.path.insert(0, './')

from strategies.sub_utils import get_unlabel_data, get_us, get_us_uc
from strategies.sub_model import get_embeddings
import arguments

args_input = arguments.get_args()
NUM_QUERY = args_input.batch
MODEL_BATCH = args_input.model_batch
UNIQ_CONTEXT = args_input.unique_context

def kmeans(n_pool, labeled_idxs, dataset, features, device, i):
    _, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, dataset)
    unlabeled_dataloader = DataLoader(unlabeled_data,
                                      collate_fn=default_data_collator,
                                      batch_size=MODEL_BATCH,
                                    )
    print('KMean querying starts.')
    print('Query {} data.'.format(NUM_QUERY))
    embeddings = get_embeddings(unlabeled_dataloader, device)
    print('Got embeddings.')
    embeddings = embeddings.numpy()

    cluster_learner = KMeans(n_clusters=NUM_QUERY*2)
    cluster_learner.fit(embeddings)
    cluster_idxs = cluster_learner.predict(embeddings)
    centers = cluster_learner.cluster_centers_[cluster_idxs]
    dis = (embeddings - centers)**2
    dis = dis.sum(axis=1)
    score_ordered_idxs = np.array([np.arange(embeddings.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(NUM_QUERY*2)])
    
    if UNIQ_CONTEXT:
        iter_i_labeled_idxs = get_us_uc(labeled_idxs, score_ordered_idxs, n_pool, features, i)
    else:
        iter_i_labeled_idxs = get_us(labeled_idxs, score_ordered_idxs, n_pool, features, i)

    return iter_i_labeled_idxs