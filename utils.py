import numpy as np
import sys

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

def get_preds(trainer_qs, unlabeled_data):
	preds, _, _ = trainer_qs.predict(unlabeled_data)
	start_logits, end_logits = preds
	return start_logits, end_logits

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)
