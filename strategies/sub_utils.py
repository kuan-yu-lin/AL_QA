import numpy as np
from itertools import combinations_with_replacement
import torch
from scipy import stats
from sklearn.metrics import pairwise_distances
import pdb
import sys
sys.path.insert(0, './')

import arguments
args_input = arguments.get_args()
LOW_RES = args_input.low_res
NUM_QUERY = args_input.batch
NUM_INIT_LB = args_input.initseed

def get_unlabel_data(n_pool, labeled_idxs, train_dataset):
    unlabeled_idxs = np.arange(n_pool)[~labeled_idxs]
    unlabeled_data = train_dataset.select(indices=unlabeled_idxs)
    return unlabeled_idxs, unlabeled_data

def get_us_uc(labeled_idxs, score_ordered_idxs, n_pool, features, iteration=0):
	ssi = set()
	uc = set()
	current_ssi = 0
	
	samples = features.select(indices=np.arange(n_pool)[labeled_idxs])
	if len(samples):
		print("\nIt's not the first query.")
		for sample in samples:
			ssi.add(sample['example_id'])
			uc.add(sample['context'])
	assert len(ssi) == len(samples), "\nThe amount of ssi from previous query is wrong. There are only {} ssi.".format(len(ssi))
	assert len(uc) == len(samples), "The amount of uc from previous query is wrong. There are only {} uc.".format(len(uc))
	print('Before filter, we already have {} instances.'.format(len(samples)))

	for soi in score_ordered_idxs:
		pool_idxs = np.zeros(len(features), dtype=bool)
		pool_idxs[soi] = True
		sample = features.select(indices=np.arange(n_pool)[pool_idxs])

		if sample[0]['example_id'] not in ssi:
			if sample[0]['context'] not in uc:
				ssi.add(sample[0]['example_id'])
				uc.add(sample[0]['context'])
				current_ssi += 1

		# check if we got enough ssi during current query
		if not iteration:
			if current_ssi == NUM_INIT_LB:
				break
		else:	
			if current_ssi == NUM_QUERY:
				break
	
	if LOW_RES:
		total = NUM_QUERY * iteration
	else:
		total = NUM_QUERY * iteration + NUM_INIT_LB
	print('We have {} unique ssi having unique context.\n'.format(len(ssi)))
	assert len(ssi) == total, "Not enough :(" 

	# select all instances belonging to unique samples
	filtered_score_ordered_idx = []
	for idxs in score_ordered_idxs:
		pool_idxs = np.zeros(len(features), dtype=bool)
		pool_idxs[idxs] = True
		sample = features.select(indices=np.arange(n_pool)[pool_idxs])
		if sample[0]['example_id'] in ssi:
			filtered_score_ordered_idx.append(idxs)

	labeled_idxs[filtered_score_ordered_idx] = True
	return np.arange(n_pool)[labeled_idxs]

def get_us(labeled_idxs, score_ordered_idxs, n_pool, features, iteration=0):
	ssi = set()
	current_ssi = 0
	
	samples = features.select(indices=np.arange(n_pool)[labeled_idxs])
	if len(samples):
		for sample in samples:
			ssi.add(sample['example_id'])
	print('Before filter, we already have {} instances.'.format(len(samples)))

	# filter out all unique samples
	for soi in score_ordered_idxs:
		pool_idxs = np.zeros(len(features), dtype=bool)
		pool_idxs[soi] = True
		sample = features.select(indices=np.arange(n_pool)[pool_idxs])

		if sample[0]['example_id'] not in ssi:
			ssi.add(sample[0]['example_id'])
			current_ssi += 1

		# check if we got enough ssi during current query
		if not iteration:
			if current_ssi == NUM_INIT_LB:
				break
		else:	
			if current_ssi == NUM_QUERY:
				break
	
	if LOW_RES:
		total = NUM_QUERY * iteration
	else:
		total = NUM_QUERY * iteration + NUM_INIT_LB
	assert len(ssi) == total, "Not enough :(" 
	
	# select all instances belonging to unique samples
	filtered_score_ordered_idx = []
	for idxs in score_ordered_idxs:
		pool_idxs = np.zeros(len(features), dtype=bool)
		pool_idxs[idxs] = True
		sample = features.select(indices=np.arange(n_pool)[pool_idxs])
		if sample[0]['example_id'] in ssi:
			filtered_score_ordered_idx.append(idxs)
	
	labeled_idxs[filtered_score_ordered_idx] = True

	# dataset = dataset.filter(lambda instance: instance['sample_id'] in set_of_selected_sample_ids)
	# num_proc=
	print('We added {} unique ssi in this query to get {} unique ssi and {} instances in total.\n'.format(current_ssi, len(ssi), len(filtered_score_ordered_idx)))
	return np.arange(n_pool)[labeled_idxs]

# kmeans ++ initialization
def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    # print('#Samps\tTotal Distance')
    print('init_centers() starts.')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll: ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    print('init_centers() done.')
    return indsAll

def class_combinations(c, n, m=np.inf):
    """ Generates an array of n-element combinations where each element is one of
    the c classes (an integer). If m is provided and m < n^c, then instead of all
    n^c combinations, m combinations are randomly sampled.

    Arguments:
        c {int} -- the number of classes
        n {int} -- the number of elements in each combination

    Keyword Arguments:
        m {int} -- the number of desired combinations (default: {np.inf})

    Returns:
        np.ndarry -- An [m x n] or [n^c x n] array of integers in [0, c)
    """

    if m < c**n:
        # randomly sample combinations
        return np.random.randint(c, size=(int(m), n))
    else:
        p_c = combinations_with_replacement(np.arange(c), n)
        return np.array(list(iter(p_c)), dtype=int)

def H(x):
    """ Compute the element-wise entropy of x

    Arguments:
        x {torch.Tensor} -- array of probabilities in (0,1)

    Keyword Arguments:
        eps {float} -- prevent failure on x == 0

    Returns:
        torch.Tensor -- H(x)
    """
    return -(x)*torch.log(x)


