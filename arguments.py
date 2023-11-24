import argparse

def decode_id(exp_id):
	[p1, p2, p3, p4] = list(exp_id)[:4]
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


def get_args():
	parser = argparse.ArgumentParser(description='Extended Deep Active Learning Toolkit')
	# basic arguments
	# parser.add_argument('--ALstrategy', '-a', default='RandomSampling', type=str, help='name of active learning strategies')
	parser.add_argument('--exp_id', required=True, type=str, help='experiment ID')
	parser.add_argument('--quota', '-q', default=2000, type=int, help='quota of active learning')
	parser.add_argument('--batch', '-b', default=500, type=int, help='batch size in one active learning iteration')
	# parser.add_argument('--dataset', '-d', default='SQuAD', type=str, help='dataset name')
	parser.add_argument('--exp_round', '-p', default=5, type=int, help='number of round of repeating the experiment')
	parser.add_argument('--max_length', default=None, type=int, help='max length of each sequence')
	# parser.add_argument('--model', '-m', default='RoBERTa', type=str, help='model name')
	parser.add_argument('--model_batch', '-c', default=8, type=int, help='batch size for training the model')
	parser.add_argument('--initseed', '-s', default=500, type=int, help='Initial pool of labeled data')
	parser.add_argument('--gpu', '-g', default=0, type=str, help='which gpu')
	parser.add_argument('--seed', '-e', default=1127, type=int, help='random seed')
	# hyper parameters
	parser.add_argument('--learning_rate', '-l', default=3e-5, type=float, help='learning rate for training')
	parser.add_argument('--train_epochs', '-o', type=int, default=3, help='number of training epochs')
	# experiment setting
	
	parser.add_argument('--dev_mode', '-x', default=False, type=bool, help='True if it runs for development.')
	# parser.add_argument('--low_resource', '-r', default=False, type=bool, help='True if it is the low resource experiment.')
	# parser.add_argument('--unique_context', '-u', default=False, type=bool, help='True if it is the experiment with unique context data.')
	args = parser.parse_args()

	MODEL_NAME, UNIQ_CONTEXT, DIST_EMBED, LOW_RES, DATA_NAME, STRATEGY_NAME,  = decode_id(args.exp_id)
	parser.add_argument('--model', default=MODEL_NAME)
	parser.add_argument('--uni_con', default=UNIQ_CONTEXT)
	parser.add_argument('--dist_embed', default=DIST_EMBED)
	parser.add_argument('--low_res', default=LOW_RES)
	parser.add_argument('--dataset', default=DATA_NAME)
	parser.add_argument('--ALstrategy', default=STRATEGY_NAME)
	return args