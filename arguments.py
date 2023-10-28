import argparse

def get_args():
	parser = argparse.ArgumentParser(description='Extended Deep Active Learning Toolkit')
	# basic arguments
	parser.add_argument('--ALstrategy', '-a', default='RandomSampling', type=str, help='name of active learning strategies')
	parser.add_argument('--quota', '-q', default=2000, type=int, help='quota of active learning')
	parser.add_argument('--batch', '-b', default=500, type=int, help='batch size in one active learning iteration')
	parser.add_argument('--dataset', '-d', default='SQuAD', type=str, help='dataset name')
	parser.add_argument('--expe_round', '-p', default=5, type=int, help='number of round of repeating the experiment')
	parser.add_argument('--max_length', default=None, type=int, help='max length of each sequence')
	parser.add_argument('--model', '-m', default='RoBERTa', type=str, help='model name')
	parser.add_argument('--model_batch', '-c', default=8, type=int, help='batch size for training the model')
	parser.add_argument('--initseed', '-s', default=500, type=int, help='Initial pool of labeled data')
	parser.add_argument('--gpu', '-g', default=0, type=str, help='which gpu')
	parser.add_argument('--seed', '-e', default=1127, type=int, help='random seed')
	# hyper parameters
	parser.add_argument('--learning_rate', '-l', default=3e-5, type=float, help='learning rate for training')
	parser.add_argument('--train_epochs', '-o', type=int, default=3, help='number of training epochs')
	# experiment setting
	parser.add_argument('--exp_id', required=True, type=int, help='experiment ID')
	parser.add_argument('--toy_exp', '-x', default=False, type=bool, help='True if it runs for development with small set of data.')
	parser.add_argument('--low_resource', '-r', default=False, type=bool, help='True if it is the low resource experiment.')
	parser.add_argument('--unique_context', '-u', default=False, type=bool, help='True if it is the experiment with unique context data.')
	
	
	args = parser.parse_args()
	return args