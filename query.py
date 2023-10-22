from torch.utils.data import DataLoader
from transformers import default_data_collator

import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import arguments
from utils import get_unlabel_data, init_centers, H, class_combinations
from model import get_prob, get_prob_dropout, get_prob_dropout_split, get_embeddings, get_grad_embeddings, get_batch_prob_dropout_split

args_input = arguments.get_args()
MODEL_BATCH = args_input.model_batch

def random_sampling_query(labeled_idxs, n):
    print('Random querying starts!')
    return np.random.choice(np.where(labeled_idxs==0)[0], n, replace=False)

def margin_sampling_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(
		unlabeled_data,
		collate_fn=default_data_collator,
		batch_size=MODEL_BATCH,
	)
    print('Margin querying starts!')
    print('Query {} data from {} unlabeled training data.'.format(n, len(unlabeled_data)))

    prob_dict = get_prob(unlabeled_dataloader, device, unlabeled_features, examples)
    print('Got probability!')
    uncertainties_dict = {}
    for idx, probs in prob_dict.items():
        if len(probs) > 1: # if prob_dict['probs'] is not 0
            sort_probs = np.sort(probs)[::-1] # This method returns a copy of the array, leaving the original array unchanged.
            uncertainties_dict[idx] = sort_probs[0] - sort_probs[1]
        elif idx:
            uncertainties_dict[idx] = np.array([0])

    sorted_uncertainties_list = sorted(uncertainties_dict.items(), key=lambda x: x[1], reverse=True)
    return unlabeled_idxs[[idx for (idx, uncertainties) in sorted_uncertainties_list[:n]]]

def least_confidence_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(
		unlabeled_data,
		collate_fn=default_data_collator,
		batch_size=MODEL_BATCH,
	)

    print('LC querying starts!')
    print('Query {} data from {} unlabeled training data.'.format(n, len(unlabeled_data)))

    prob_dict = get_prob(unlabeled_dataloader, device, unlabeled_features, examples)
    print('Got probability!')

    confidence_dict = {}
    for idx, probs in prob_dict.items():
        if len(probs) > 1: # if prob_dict['probs'] is not 0
            confidence_dict[idx] = max(probs)
        elif idx:
            confidence_dict[idx] = np.array([0])

    sorted_confidence_list = sorted(confidence_dict.items(), key=lambda x: x[1])
    return unlabeled_idxs[[idx for (idx, confidence) in sorted_confidence_list[:n]]]

def var_ratio_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(
		unlabeled_data,
		collate_fn=default_data_collator,
		batch_size=MODEL_BATCH,
	)
    
    print('Var Ratio querying starts!')
    print('Query {} data from {} unlabeled training data.'.format(n, len(unlabeled_data)))

    prob_dict = get_prob(unlabeled_dataloader, device, unlabeled_features, examples)
    print('Got probability!')
    confidence_dict = {}
    for idx, probs in prob_dict.items():
        if len(probs) > 1: # if prob_dict['probs'] is not 0
            confidence_dict[idx] = 1.0 - max(probs)
        elif idx:
            confidence_dict[idx] = np.array([0])

    sorted_confidence_list = sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)
    return unlabeled_idxs[[idx for (idx, confidence) in sorted_confidence_list[:n]]]

def entropy_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(
		unlabeled_data,
		collate_fn=default_data_collator,
		batch_size=MODEL_BATCH,
	)
    
    print('Entropy querying starts!')
    print('Query {} data from {} unlabeled training data.'.format(n, len(unlabeled_data)))

    prob_dict = get_prob(unlabeled_dataloader, device, unlabeled_features, examples)
    print('Got probability!')
    entropy_dict = {}
    for idx, probs in prob_dict.items():
        if len(probs) > 1: # if prob_dict['probs'] is not 0
            log_probs = np.log(probs)
            entropy_dict[idx] = (probs*log_probs).sum()
        elif idx:
            entropy_dict[idx] = np.array([0])
    sorted_entropy_list = sorted(entropy_dict.items(), key=lambda x: x[1])
    return unlabeled_idxs[[idx for (idx, entropy) in sorted_entropy_list[:n]]]

def margin_sampling_dropout_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(
		unlabeled_data,
		collate_fn=default_data_collator,
		batch_size=MODEL_BATCH,
	)
    
    print('Margin dropout querying starts!')
    print('Query {} data from {} unlabeled training data.'.format(n, len(unlabeled_data)))

    prob_dict = get_prob_dropout(unlabeled_dataloader, device, unlabeled_features, examples, n_drop=10)
    print('Got probability!')
    uncertainties_dict = {}
    for idx, probs in prob_dict.items():
        if len(probs) > 1: # if prob_dict['probs'] is not 0
            sort_probs = np.sort(probs)[::-1] # This method returns a copy of the array, leaving the original array unchanged.
            uncertainties_dict[idx] = sort_probs[0] - sort_probs[1]
        elif idx:
            uncertainties_dict[idx] = np.array([0])

    sorted_uncertainties_list = sorted(uncertainties_dict.items(), key=lambda x: x[1], reverse=True)
    return unlabeled_idxs[[idx for (idx, uncertainties) in sorted_uncertainties_list[:n]]]

def least_confidence_dropout_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(
		unlabeled_data,
		collate_fn=default_data_collator,
		batch_size=MODEL_BATCH,
	)
    
    print('LC dropout querying starts!')
    print('Query {} data from {} unlabeled training data.'.format(n, len(unlabeled_data)))
    
    prob_dict = get_prob_dropout(unlabeled_dataloader, device, unlabeled_features, examples, n_drop=10)
    print('Got probability!')

    confidence_dict = {}
    for idx, probs in prob_dict.items():
        if len(probs) > 1: # if prob_dict['probs'] is not 0
            confidence_dict[idx] = max(probs)
        elif idx:
            confidence_dict[idx] = np.array([0])

    sorted_confidence_list = sorted(confidence_dict.items(), key=lambda x: x[1])
    return unlabeled_idxs[[idx for (idx, confidence) in sorted_confidence_list[:n]]]

def entropy_dropout_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(
		unlabeled_data,
		collate_fn=default_data_collator,
		batch_size=MODEL_BATCH,
	)
    
    print('Entropy dropout querying starts!')
    print('Query {} data from {} unlabeled training data.'.format(n, len(unlabeled_data)))
    
    prob_dict = get_prob_dropout(unlabeled_dataloader, device, unlabeled_features, examples, n_drop=10)
    print('Got probability!')
    entropy_dict = {}
    for idx, probs in prob_dict.items():
        if len(probs) > 1: # if prob_dict['probs'] is not 0
            log_probs = np.log(probs)
            entropy_dict[idx] = (probs*log_probs).sum()
        elif idx:
            entropy_dict[idx] = np.array([0])
    sorted_entropy_list = sorted(entropy_dict.items(), key=lambda x: x[1])
    return unlabeled_idxs[[idx for (idx, entropy) in sorted_entropy_list[:n]]]

def bald_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(
        unlabeled_data,
        collate_fn=default_data_collator,
        batch_size=MODEL_BATCH,
    )
    print('BALD querying starts!')
    print('Query {} data from {} unlabeled training data.'.format(n, len(unlabeled_data)))
    
    probs = get_prob_dropout_split(unlabeled_dataloader, device, unlabeled_features, examples, n_drop=10)
    print('Got probability!')
    probs_mean = probs.mean(0)
    entropy1 = (-probs_mean*torch.log(probs_mean)).sum(1)
    entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
    uncertainties = entropy2 - entropy1
    # I(y;W | x) = H1 - H2 = H(y|x) - E_w[H(y|x,W)]
    return unlabeled_idxs[uncertainties.sort()[1][:n]]

def mean_std_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(
  		unlabeled_data,
        collate_fn=default_data_collator,
        batch_size=MODEL_BATCH,
    )
    print('Mean STD querying starts!')
    print('Query {} data from {} unlabeled training data.'.format(n, len(unlabeled_data)))
    
    probs = get_prob_dropout_split(unlabeled_dataloader, device, unlabeled_features, examples, n_drop=10).numpy()
    print('Got probability!')
    sigma_c = np.std(probs, axis=0)
    uncertainties = torch.from_numpy(np.mean(sigma_c, axis=-1)) # use tensor.sort() will sort the data and produce sorted indexes
    return unlabeled_idxs[uncertainties.sort(descending=True)[1][:n]]

def kmeans_query(n_pool, labeled_idxs, train_dataset, device, n):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    # unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(unlabeled_data,
                                      collate_fn=default_data_collator,
                                      batch_size=MODEL_BATCH,
                                    )
    print('KMean querying starts!')
    print('Query {} data.'.format(n))
    
    embeddings = get_embeddings(unlabeled_dataloader, device)
    print('Got embeddings!')
    embeddings = embeddings.numpy()

    cluster_learner = KMeans(n_clusters=n)
    cluster_learner.fit(embeddings)
    cluster_idxs = cluster_learner.predict(embeddings)
    centers = cluster_learner.cluster_centers_[cluster_idxs]
    dis = (embeddings - centers)**2
    dis = dis.sum(axis=1)
    q_idxs = np.array([np.arange(embeddings.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])

    return unlabeled_idxs[q_idxs]

def kcenter_greedy_query(n_pool, labeled_idxs, train_dataset, device, n):
    labeled_idxs_in_query = labeled_idxs.copy()
    # train_data = train_dataset
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=default_data_collator,
                                  batch_size=MODEL_BATCH,
                                )
    print('KCenter greedy querying starts!')
    print('Query {} data.'.format(n))
    
    embeddings = get_embeddings(train_dataloader, device)
    print('Got embeddings!')
    embeddings = embeddings.numpy()

    dist_mat = np.matmul(embeddings, embeddings.transpose())
    sq = np.array(dist_mat.diagonal()).reshape(len(labeled_idxs_in_query), 1)
    dist_mat *= -2
    dist_mat += sq
    dist_mat += sq.transpose()
    dist_mat = np.sqrt(dist_mat)

    mat = dist_mat[~labeled_idxs_in_query, :][:, labeled_idxs_in_query]

    for i in tqdm(range(n), ncols=100):
        mat_min = mat.min(axis=1)
        q_idx_ = mat_min.argmax()
        q_idx = np.arange(n_pool)[~labeled_idxs_in_query][q_idx_]
        labeled_idxs_in_query[q_idx] = True
        mat = np.delete(mat, q_idx_, 0)
        mat = np.append(mat, dist_mat[~labeled_idxs_in_query, q_idx][:, None], axis=1)
        
    return np.arange(n_pool)[(labeled_idxs ^ labeled_idxs_in_query)]

def kcenter_greedy_PCA_query(n_pool, labeled_idxs, train_dataset, device, n):
    labeled_idxs_in_query = labeled_idxs.copy()
    # train_data = train_dataset
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=default_data_collator,
                                  batch_size=MODEL_BATCH,
                                )
    print('KCenter greedy PCA querying starts!')
    print('Query {} data.'.format(n))

    embeddings = get_embeddings(train_dataloader, device)
    print('Got embeddings!')
    embeddings = embeddings.numpy()
    dist_mat = np.matmul(embeddings, embeddings.transpose())

    if len(embeddings[0]) > 50:
        pca = PCA(n_components=50)
        embeddings = pca.fit_transform(embeddings)
    embeddings = embeddings.astype(np.float16)

    sq = np.array(dist_mat.diagonal()).reshape(len(labeled_idxs_in_query), 1)
    dist_mat *= -2
    dist_mat += sq
    dist_mat += sq.transpose()
    dist_mat = np.sqrt(dist_mat)

    mat = dist_mat[~labeled_idxs_in_query, :][:, labeled_idxs_in_query]

    for i in tqdm(range(n), ncols=100):
        mat_min = mat.min(axis=1)
        q_idx_ = mat_min.argmax()
        q_idx = np.arange(n_pool)[~labeled_idxs_in_query][q_idx_]
        labeled_idxs_in_query[q_idx] = True
        mat = np.delete(mat, q_idx_, 0)
        mat = np.append(mat, dist_mat[~labeled_idxs_in_query, q_idx][:, None], axis=1)
        
    return np.arange(n_pool)[(labeled_idxs ^ labeled_idxs_in_query)]

def badge_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(unlabeled_data,
                                      collate_fn=default_data_collator,
                                      batch_size=MODEL_BATCH,
                                    )
    print('BADGE querying starts!')
    print('Query {} data from {} unlabeled training data.'.format(n, len(unlabeled_data)))

    gradEmbedding = get_grad_embeddings(unlabeled_dataloader, device, unlabeled_features, examples)
    print('Got embeddings!')
    chosen = init_centers(gradEmbedding, n)
    return unlabeled_idxs[chosen]

def batch_bald_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    # unlabeled_features = train_features.select(unlabeled_idxs)
    print('BatchBALD querying starts!')
    print('Query {} data from {} unlabeled training data.'.format(n, len(unlabeled_data)))
    m = 1e4  # number of MC samples for label combinations
    # if LOW_RES:
    #     num_sub_pool = len(unlabeled_data)
    # else:
    #     num_sub_pool = 500  # number of datapoints in the subpool from which we acquire
    num_sub_pool = 500  # number of datapoints in the subpool from which we acquire

    c = 10
    k = 10
    processing_batch_size = 8 # Model_batch=8 or 128

    # performing BatchBALD on the whole pool is very expensive, so we take a random subset of the pool.
    num_extra = len(unlabeled_data) - num_sub_pool

    if num_extra > 0:
        sub_pool_data, _ = torch.utils.data.random_split(unlabeled_data, [num_sub_pool, num_extra])
        sub_pool_idxs = np.array(sub_pool_data.indices)
    else:
        # even if we don't have enough data left to split, we still need to
        # call random_splot to avoid messing up the indexing later on
        sub_pool_data, _ = torch.utils.data.random_split(unlabeled_data, [len(unlabeled_data), 0])
        sub_pool_idxs = np.array(sub_pool_data.indices)

    # forward pass on the pool once to get class probabilities for each x
    pool_loader = torch.utils.data.DataLoader(sub_pool_data,
        batch_size=processing_batch_size, pin_memory=True, shuffle=False, collate_fn=default_data_collator)
    pool_features = train_features.select(sub_pool_idxs)

    pool_p_y = get_batch_prob_dropout_split(pool_loader, device, pool_features, examples, n_drop=k).permute(1, 0, 2) 
    # pool_p_y.shape = n * k * c = 500 * 3 * 200
    # this only need to be calculated once so we pull it out of the loop
    H2 = (H(pool_p_y).sum(axis=(1,2))/k)

    # get all class combinations
    c_1_to_n = class_combinations(c, n, m) # TODO: change from 'n' to 'total_query' # TODO: test1- change c to k
    # print('c_1_to_n.shape:', c_1_to_n.shape) # (10000, 138) ? # tensor of size [m * n]

    # tensor of size [m * k]
    p_y_1_to_n_minus_1 = None

    # store the indices of the chosen datapoints in the subpool
    best_sub_local_indices = []

    # create a mask to keep track of which indices we've chosen
    remaining_indices = torch.ones(len(sub_pool_data), dtype=bool)
    # remaining_indices = torch.ones(total_query, dtype=bool)
    for batch_n in range(n): # TODO: change from 'n' to 'total_query'
        # tensor of size [N * m * l] # [500, 10000, 3]
        p_y_n = pool_p_y[:, c_1_to_n[:, batch_n], :] # wierd here, bc k was < c # TODO: test1
        # p_y_n = pool_p_y[:, :, c_1_to_n[:, batch_n]] # try with this
        # tensor of size [N * m * k]   
        if p_y_1_to_n_minus_1 == None:
            p_y_1_to_n = p_y_n
            # print(1)
        elif torch.tensor(0) in p_y_1_to_n_minus_1:
            p_y_1_to_n = p_y_n
            # print(2)
        else:
            p_y_1_to_n = torch.einsum('mk,pmk->pmk', p_y_1_to_n_minus_1, p_y_n)
            # print(3)

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
    assert len(set(best_global_indices)) == n, f'there are only {len(set(best_global_indices))} unique idx in q_idx'
    return best_global_indices