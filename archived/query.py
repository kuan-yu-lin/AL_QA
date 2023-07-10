from torch.utils.data import DataLoader
from transformers import default_data_collator

import numpy as np
import torch
import collections
from tqdm.auto import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from utils import get_unlabel_data, init_centers
from model import get_prob, get_prob_dropout, get_prob_dropout_split, get_embeddings, get_grad_embeddings

def random_sampling_query(labeled_idxs, n):
    return np.random.choice(np.where(labeled_idxs==0)[0], n, replace=False)

def margin_sampling_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n, rd):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(unlabeled_data,
                                      shuffle=False,
                                      collate_fn=default_data_collator,
                                      batch_size=8,
                                    )
    print('Margin querying starts!')
    prob_dict = get_prob(unlabeled_dataloader, device, unlabeled_features, examples, rd)
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

def least_confidence_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n, rd):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(unlabeled_data,
                                      shuffle=False,
                                      collate_fn=default_data_collator,
                                      batch_size=8,
                                    )
    print('LC querying starts!')
    prob_dict = get_prob(unlabeled_dataloader, device, unlabeled_features, examples, rd)
    print('Got probability!')

    confidence_dict = {}
    for idx, probs in prob_dict.items():
        if len(probs) > 1: # if prob_dict['probs'] is not 0
            confidence_dict[idx] = max(probs)
        elif idx:
            confidence_dict[idx] = np.array([0])

    sorted_confidence_list = sorted(confidence_dict.items(), key=lambda x: x[1])
    return unlabeled_idxs[[idx for (idx, confidence) in sorted_confidence_list[:n]]]

def var_ratio_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n, rd):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(unlabeled_data,
                                      shuffle=False,
                                      collate_fn=default_data_collator,
                                      batch_size=8,
                                    )
    print('Var Ratio querying starts!')
    prob_dict = get_prob(unlabeled_dataloader, device, unlabeled_features, examples, rd)
    print('Got probability!')
    confidence_dict = {}
    for idx, probs in prob_dict.items():
        if len(probs) > 1: # if prob_dict['probs'] is not 0
            confidence_dict[idx] = 1.0 - max(probs)
        elif idx:
            confidence_dict[idx] = np.array([0])

    sorted_confidence_list = sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)
    return unlabeled_idxs[[idx for (idx, confidence) in sorted_confidence_list[:n]]]

def entropy_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n, rd):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(unlabeled_data,
                                      shuffle=False,
                                      collate_fn=default_data_collator,
                                      batch_size=8,
                                    )
    print('Entropy querying starts!')
    prob_dict = get_prob(unlabeled_dataloader, device, unlabeled_features, examples, rd)
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

def margin_sampling_dropout_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n, rd):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(unlabeled_data,
                                      shuffle=False,
                                      collate_fn=default_data_collator,
                                      batch_size=8,
                                    )
    print('Margin dropout querying starts!')
    prob_dict = get_prob_dropout(unlabeled_dataloader, device, unlabeled_features, examples, rd)
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

def least_confidence_dropout_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n, rd):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(unlabeled_data,
                                      shuffle=False,
                                      collate_fn=default_data_collator,
                                      batch_size=8,
                                    )
    print('LC dropout querying starts!')
    prob_dict = get_prob_dropout(unlabeled_dataloader, device, unlabeled_features, examples, rd)
    print('Got probability!')

    confidence_dict = {}
    for idx, probs in prob_dict.items():
        if len(probs) > 1: # if prob_dict['probs'] is not 0
            confidence_dict[idx] = max(probs)
        elif idx:
            confidence_dict[idx] = np.array([0])

    sorted_confidence_list = sorted(confidence_dict.items(), key=lambda x: x[1])
    return unlabeled_idxs[[idx for (idx, confidence) in sorted_confidence_list[:n]]]

def entropy_dropout_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n, rd):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(unlabeled_data,
		                              shuffle=False,
	                    	          collate_fn=default_data_collator,
		                              batch_size=8,
	                                )
    print('Entropy dropout querying starts!')
    prob_dict = get_prob_dropout(unlabeled_dataloader, device, unlabeled_features, examples, rd)
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

def bayesian_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n, rd):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(unlabeled_data,
                                      shuffle=False,
                                      collate_fn=default_data_collator,
                                      batch_size=8,
                                    )
    print('BALD querying starts!')
    probs = get_prob_dropout_split(unlabeled_dataloader, device, unlabeled_features, examples, rd)
    print('Got probability!')
    probs_mean = probs.mean(0)
    entropy1 = (-probs_mean*torch.log(probs_mean)).sum(1)
    entropy2 = (-probs*torch.log(probs)).sum(2).mean(0)
    uncertainties = entropy2 - entropy1
    # later on, we can use batch
    return unlabeled_idxs[uncertainties.sort()[1][:n]]

def mean_std_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n, rd):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(unlabeled_data,
                                      shuffle=False,
                                      collate_fn=default_data_collator,
                                      batch_size=8,
                                    )
    print('Mean STD querying starts!')
    probs = get_prob_dropout_split(unlabeled_dataloader, device, unlabeled_features, examples, rd).numpy()
    print('Got probability!')
    sigma_c = np.std(probs, axis=0)
    uncertainties = torch.from_numpy(np.mean(sigma_c, axis=-1)) # use tensor.sort() will sort the data and produce sorted indexes
    return unlabeled_idxs[uncertainties.sort(descending=True)[1][:n]]

def kmeans_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n, rd):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(unlabeled_data,
                                      shuffle=False,
                                      collate_fn=default_data_collator,
                                      batch_size=8,
                                    )
    print('KMean querying starts!')
    embeddings = get_embeddings(unlabeled_dataloader, device, unlabeled_features, examples, rd)
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

def kcenter_greedy_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n, rd):
    labeled_idxs_in_query = labeled_idxs.copy()
    # train_data = train_dataset
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=False,
                                  collate_fn=default_data_collator,
                                  batch_size=8,
                                )
    print('KCenter greedy querying starts!')
    embeddings = get_embeddings(train_dataloader, device, train_features, examples, rd)
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

def kcenter_greedy_PCA_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n, rd):
    labeled_idxs_in_query = labeled_idxs.copy()
    # train_data = train_dataset
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=False,
                                  collate_fn=default_data_collator,
                                  batch_size=8,
                                )
    print('KCenter greedy PCA querying starts!')
    embeddings = get_embeddings(train_dataloader, device, train_features, examples, rd)
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

def badge_query(n_pool, labeled_idxs, train_dataset, train_features, examples, device, n, rd):
    unlabeled_idxs, unlabeled_data = get_unlabel_data(n_pool, labeled_idxs, train_dataset)
    unlabeled_features = train_features.select(unlabeled_idxs)
    unlabeled_dataloader = DataLoader(unlabeled_data,
                                      shuffle=False,
                                      collate_fn=default_data_collator,
                                      batch_size=8,
                                    )
    print('BADGE querying starts!')
    gradEmbedding = get_grad_embeddings(unlabeled_dataloader, device, unlabeled_features, examples, rd)
    print('Got embeddings!')
    chosen = init_centers(gradEmbedding, n)
    return unlabeled_idxs[chosen]

def loss_prediction_query():
    pass

def ceal_query():
    pass
