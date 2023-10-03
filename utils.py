from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering
import numpy as np
from scipy import stats
from sklearn.metrics import pairwise_distances
import pdb
import sys
from collections import Counter
import string
import re

import arguments

CACHE_DIR = '/mount/arbeitsdaten31/studenten1/linku/.cache'
args_input = arguments.get_args()
LOW_RES = args_input.low_resource

class Logger(object):
	def __init__(self, filename="Default.log"):
		self.terminal = sys.stdout
		self.log = open(filename, "a")

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)

	def flush(self):
		pass


def get_aubc(quota, bsize, resseq):
	# it is equal to use np.trapz for calculation
	ressum = 0.0
	if quota % bsize == 0:
		for i in range(len(resseq)-1):
			ressum = ressum + (resseq[i+1] + resseq[i]) * bsize / 2

	else:
		for i in range(len(resseq)-2):
			ressum = ressum + (resseq[i+1] + resseq[i]) * bsize / 2
		k = quota % bsize
		ressum = ressum + ((resseq[-1] + resseq[-2]) * k / 2)
	ressum = round(ressum / quota,3)
	
	return ressum


def get_mean_stddev(datax):
	return round(np.mean(datax),4),round(np.std(datax),4)


def get_unlabel_data(n_pool, labeled_idxs, train_dataset):
    unlabeled_idxs = np.arange(n_pool)[~labeled_idxs]
    unlabeled_data = train_dataset.select(indices=unlabeled_idxs)
    return unlabeled_idxs, unlabeled_data


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def get_model(m):
	if m.lower() == 'bert':
		return 'bert-base-uncased'
	elif m.lower() == 'bertlarge':
		return 'bert-large-uncased'
	elif m.lower() == 'roberta':
		return 'roberta-base'
	elif m.lower() == 'robertalarge':
		return 'roberta-large'


# kmeans ++ initialization
def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll: ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll


def load_dataset_mrqa(d):
	'''
	return train_set, val_set
	'''
	data = load_dataset("mrqa", cache_dir=CACHE_DIR)
	if d == 'squad':
		# the first to 86588th in train set
		# the first to 10507th in val set
		squad_train = data['train'].select(range(86588))
		squad_val = data['validation'].select(range(10507))
		for t in squad_train: assert t['subset'] == 'SQuAD', 'Please select corrrect train data for SQuAD.'
		for v in squad_val: assert v['subset'] == 'SQuAD', 'Please select corrrect validation data for SQuAD.'
		return squad_train, squad_val
	elif d == 'newsqa':
		# the 86589th to 160748th in train set
		# the 10508th to 14719th in val set
		data_set = data['train'].select(range(86588, 160748))
		newsqa_train = data_set.shuffle(1127).select(range(10000))
		newsqa_val = data['validation'].select(range(10507, 14719))
		for t in newsqa_train: assert t['subset'] == 'NewsQA', 'Please select corrrect train data for NewQA.'
		for v in newsqa_val: assert v['subset'] == 'NewsQA', 'Please select corrrect validation data for NewQA.'
		return newsqa_train, newsqa_val
	elif d == 'searchqa':
		# the 222437th to 339820th in train set
		# the 22505th to 39484th in val set
		data_set = data['train'].select(range(222436, 339820))
		searchqa_train = data_set.shuffle(1127).select(range(10000))
		searchqa_val = data['validation'].select(range(22504, 39484))	
		for t in searchqa_train: assert t['subset'] == 'SearchQA', 'Please select corrrect train data for SearchQA.'
		for v in searchqa_val: assert v['subset'] == 'SearchQA', 'Please select corrrect validation data for SearchQA.'
		return searchqa_train, searchqa_val
	elif d == 'bioasq':
		# the first to the 1504th in the test set
		sub = data['test'].select(range(1504))
		len_sub_val = len(sub) // 10
		bioasq_train = sub.select(range(len_sub_val, len(sub)))
		bioasq_val = sub.select(range(len_sub_val))
		for t in bioasq_train: assert t['subset'] == 'BioASQ', 'Please select corrrect train data for BioASQ.'
		for v in bioasq_val: assert v['subset'] == 'BioASQ', 'Please select corrrect validation data for BioASQ.'
		return bioasq_train, bioasq_val
	elif d == 'textbookqa':
		# the 8131st to 9633rd
		sub = data['test'].select(range(8130, 9633))
		len_sub_val = len(sub) // 10
		textbookqa_train = sub.select(range(len_sub_val, len(sub)))
		textbookqa_val = sub.select(range(len_sub_val)) 
		for t in textbookqa_train: assert t['subset'] == 'TextbookQA', 'Please select corrrect train data for TextbookQA.'
		for v in textbookqa_val: assert v['subset'] == 'TextbookQA', 'Please select corrrect validation data for TextbookQA.'
		return textbookqa_train, textbookqa_val
	elif d == 'drop': # Discrete Reasoning Over Paragraphs
		# the 1505th to 3007th in test set
		sub = data['test'].select(range(1504, 3007))
		len_sub_val = len(sub) // 10
		drop_train = sub.select(range(len_sub_val, len(sub)))
		drop_val = sub.select(range(len_sub_val))
		for t in drop_train: assert t['subset'] == 'DROP', 'Please select corrrect train data for DROP.'
		for v in drop_val: assert v['subset'] == 'DROP', 'Please select corrrect validation data for DROP.'
		return drop_train, drop_val
	

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(theoretical_answers, predicted_answers, skip_no_answer=False):
    '''
	theoretical_answers, datatype=dict
	{strings of id: list of ground truth answers}
	predicted_answers, datatype=dict
	{strings of id: strings of prediction text}
	'''
    f1 = exact_match = total = 0
    for qid, ground_truths in theoretical_answers.items():
        if qid not in predicted_answers:
            if not skip_no_answer:
                message = 'Unanswered question %s will receive score 0.' % qid
                print(message)
                total += 1
            continue
        total += 1
        prediction = predicted_answers[qid]
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}


def save_model(device, pretrain_dir, strategy_dir):
    '''
    Copy and save model from pretrain_models to current trained models.
    '''
    pretrain_model = AutoModelForQuestionAnswering.from_pretrained(pretrain_dir).to(device)
    model_to_save = pretrain_model.module if hasattr(pretrain_model, 'module') else pretrain_model 
    model_to_save.save_pretrained(strategy_dir)
    
def get_unique_sample(labeled_idxs, q_idxs, n_pool, train_features, iteration=0):
	if LOW_RES:
		num_set_query_i = NUM_QUERY * iteration
		print('num_set_query_i', num_set_query_i)
	elif iteration == 0:
		num_set_query_i = NUM_INIT_LB
	else:
		num_set_query_i = NUM_QUERY * iteration + NUM_INIT_LB

	difference_i = 0
	
	while True:
		labeled_idxs[q_idxs[:NUM_QUERY + difference_i]] = True	# get first num_query, e.g. 50
		run_i_labeled_idxs = np.arange(n_pool)[labeled_idxs]
		run_i_samples = train_features.select(indices=run_i_labeled_idxs)
		num_set_ex_id_i = len(set(run_i_samples['example_id']))
		print('number of unique example id:', num_set_ex_id_i)

		difference_i = num_set_query_i - num_set_ex_id_i
		print('difference_i', difference_i)
		
		if difference_i == 0:
			break
	
	return run_i_labeled_idxs

def get_unique_sample_and_context(labeled_idxs, q_idxs, n_pool, train_features, iteration=0):
	if LOW_RES:
		num_set_query_i = NUM_QUERY * iteration
		print('num_set_query_i', num_set_query_i)
	elif iteration == 0:
		num_set_query_i = NUM_INIT_LB
	else:
		num_set_query_i = NUM_QUERY * iteration + NUM_INIT_LB

	difference_i = 0
	
	while True:
		labeled_idxs[q_idxs[:NUM_QUERY + difference_i]] = True	# get first num_query, e.g. 50
		run_i_labeled_idxs = np.arange(n_pool)[labeled_idxs]
		run_i_samples = train_features.select(indices=run_i_labeled_idxs)
		num_set_co_id_i = len(set(run_i_samples['context_id']))
		print('number of unique context id:', num_set_co_id_i)
		num_set_ex_id_i = len(set(run_i_samples['example_id']))
		print('number of unique example id:', num_set_ex_id_i)

		difference_i = max((num_set_query_i - num_set_co_id_i), (num_set_query_i - num_set_ex_id_i))
		print('difference_i', difference_i)
		
		if difference_i <= 0:
			break
	
	return run_i_labeled_idxs
