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
NUM_QUERY = args_input.batch
NUM_INIT_LB = args_input.initseed

def sub_decode_id():
	args_input = arguments.get_args()
	[p1, p2, p3, p4] = list(str(args_input.exp_id))[:4]
	UNIQ_CONTEXT = False
	LOW_RES = False
	MODEL_NAME = 'RoBERTa'

	if p1 == '2':
		MODEL_NAME = 'BERT'
	elif p1 == '3':
		MODEL_NAME = 'RoBERTaLarge'
	elif p1 == '4':
		MODEL_NAME = 'BERTLarge'
	elif p1 == '5':
		UNIQ_CONTEXT = True

	if p2 == '2': LOW_RES = True
	
	if p3 == '1':
		DATA_NAME = 'SQuAD'
	elif p3 == '2':
		DATA_NAME = 'BioASQ'
	elif p3 == '3':
		DATA_NAME = 'DROP'
	elif p3 == '4':
		DATA_NAME = 'TextbookQA'
	elif p3 == '5':
		DATA_NAME = 'NewsQA'
	elif p3 == '6':
		DATA_NAME = 'SearchQA'
	elif p3 == '7':
		DATA_NAME = 'NaturalQuestions'
	
	if p4 == 'a':
		STRATEGY_NAME = 'RandomSampling'
	elif p4 == 'b':
		STRATEGY_NAME = 'MarginSampling'
	elif p4 == 'c':
		STRATEGY_NAME = 'LeastConfidence'
	elif p4 == 'd':
		STRATEGY_NAME = 'EntropySampling'
	elif p4 == 'e':
		STRATEGY_NAME = 'MarginSamplingDropout'
	elif p4 == 'f':
		STRATEGY_NAME = 'LeastConfidenceDropout'
	elif p4 == 'g':
		STRATEGY_NAME = 'EntropySamplingDropout'
	elif p4 == 'h':
		STRATEGY_NAME = 'KMeansSampling'
	elif p4 == 'i':
		STRATEGY_NAME = 'KCenterGreedy'
	elif p4 == 'j':
		STRATEGY_NAME = 'MeanSTD'
	elif p4 == 'k':
		STRATEGY_NAME = 'BALDDropout'
	elif p4 == 'l':
		STRATEGY_NAME = 'BadgeSampling'
	elif p4 == 'm':
		STRATEGY_NAME = 'BatchBALD'
	
	return LOW_RES, DATA_NAME, STRATEGY_NAME, MODEL_NAME, UNIQ_CONTEXT

LOW_RES, _, _, _, _ = sub_decode_id()

def get_unlabel_data(n_pool, labeled_idxs, train_dataset):
    unlabeled_idxs = np.arange(n_pool)[~labeled_idxs]
    unlabeled_data = train_dataset.select(indices=unlabeled_idxs)
    return unlabeled_idxs, unlabeled_data

def get_us_uc(labeled_idxs, score_ordered_idxs, n_pool, features, iteration=0):
	ssi = set()
	uc = set()
	
	samples = features.select(indices=np.arange(n_pool)[labeled_idxs])
	if len(samples):
		print("\nIt's not the first query.")
		for sample in samples:
			ssi.add(sample['example_id'])
			uc.add(sample['context'])
	assert len(ssi) == len(samples), "\nThe amount of ssi from previous query is wrong. There are only {} ssi.".format(len(ssi))
	assert len(uc) == len(samples), "The amount of uc from previous query is wrong. There are only {} uc.".format(len(uc))
	print('Before filter, we already have {} instances.'.format(len(samples)))

	filtered_score_ordered_idx = []
	for soi in score_ordered_idxs:
		pool_idxs = np.zeros(len(features), dtype=bool)
		pool_idxs[soi] = True
		sample = features.select(indices=np.arange(n_pool)[pool_idxs])

		if sample[0]['example_id'] not in ssi:
			if sample[0]['context'] not in uc:
				ssi.add(sample[0]['example_id'])
				uc.add(sample[0]['context'])
				filtered_score_ordered_idx.append(soi)

		if not iteration:
			if len(filtered_score_ordered_idx) == NUM_INIT_LB:
				break
		else:	
			if len(filtered_score_ordered_idx) == NUM_QUERY:
				break
	
	print('We have {} unique ssi having unique context.\n'.format(len(ssi)))
	if not iteration:
		assert len(filtered_score_ordered_idx) == NUM_INIT_LB, "Not enough :("
	else:	
		assert len(filtered_score_ordered_idx) == NUM_QUERY, "Not enough :("

	labeled_idxs[filtered_score_ordered_idx] = True
	return np.arange(n_pool)[labeled_idxs]

def get_us(labeled_idxs, score_ordered_idxs, n_pool, features, iteration=0):
	ssi = set()
	current_ssi = set()
	
	samples = features.select(indices=np.arange(n_pool)[labeled_idxs])
	if len(samples):
		for sample in samples:
			ssi.add(sample['example_id'])
	print('Before filter, we already have {} instances.'.format(len(samples)))

	filtered_score_ordered_idx = []
	for i, soi in enumerate(score_ordered_idxs):
		pool_idxs = np.zeros(len(features), dtype=bool)
		pool_idxs[soi] = True
		sample = features.select(indices=np.arange(n_pool)[pool_idxs])

		if sample[0]['example_id'] not in ssi:
			ssi.add(sample[0]['example_id'])
			current_ssi.add(sample[0]['example_id'])
			filtered_score_ordered_idx.append(soi)

		# check if we got enough ssi during current query
		if not iteration:
			if len(current_ssi) == NUM_INIT_LB:
				sliced = i + 1
				break
		else:	
			if len(current_ssi) == NUM_QUERY:
				sliced = i + 1
				break
	
	if LOW_RES:
		total = NUM_QUERY * iteration
	else:
		total = NUM_QUERY * iteration + NUM_INIT_LB
	assert len(ssi) == total, "Not enough :(" 

	for idxs in score_ordered_idxs[sliced:]:
		pool_idxs = np.zeros(len(features), dtype=bool)
		pool_idxs[idxs] = True
		sample = features.select(indices=np.arange(n_pool)[pool_idxs])
		if sample[0]['example_id'] in ssi:
			filtered_score_ordered_idx.append(idxs)
	
	# labeled_idxs[score_ordered_idxs[:sliced]] = True
	labeled_idxs[filtered_score_ordered_idx] = True

	# dataset = dataset.filter(lambda instance: instance['sample_id'] in set_of_selected_sample_ids)
	# num_proc=
	print('We added {} unique ssi in this query to get {} unique ssi and {} instances in total.\n'.format(len(current_ssi), len(ssi), len(filtered_score_ordered_idx)))
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


