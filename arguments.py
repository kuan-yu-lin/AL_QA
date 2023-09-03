import argparse

def get_args():
	parser = argparse.ArgumentParser(description='Extended Deep Active Learning Toolkit')
	# basic arguments
	parser.add_argument('--ALstrategy', '-a', default='RandomSampling', type=str, help='name of active learning strategies')
	parser.add_argument('--quota', '-q', default=100, type=int, help='quota of active learning')
	parser.add_argument('--batch', '-b', default=35, type=int, help='batch size in one active learning iteration')
	parser.add_argument('--dataset_name', '-d', default='SQuAD', type=str, help='dataset name')
	parser.add_argument('--iteration', '-t', default=3, type=int, help='time of repeat the experiment')
	
	# parser.add_argument('--data_path', type=str, default='./../data', help='Path to where the data is')
	# parser.add_argument('--out_path', type=str, default='./../results', help='Path to where the output log will be')
	# TODO: path to save model
	parser.add_argument('--max_length', type=int, default=None, help='Max length of each sequence')
	# parser.add_argument('--log_name', type=str, default='test.log', help='middle outputs')
	#parser.add_argument('--help', '-h', default=False, action='store_true', help='verbose')
	parser.add_argument('--cuda', action='store_true', help='If training is to be done on a GPU')
	parser.add_argument('--model', '-m', default='Bert', type=str, help='model name')
	parser.add_argument('--initseed', '-s', default = 100, type = int, help = 'Initial pool of labeled data')
	parser.add_argument('--gpu', '-g', default = 0, type = str, help = 'which gpu')
	parser.add_argument('--seed', '-e', default=1127, type=int, help='random seed')
	# new args
	parser.add_argument('--model_batch', '-c', default=8, type=int, help='batch size for training the model')
	parser.add_argument('--toy_exp', '-x', default=False, type=bool, help='True if it runs for development with small set of data.')
	parser.add_argument('--learning_rate', '-l', default=1e-4, type=float, help='learning rate for training')
	# lpl
	# parser.add_argument('--lpl_epoches', type=int, default=20, help='lpl epoch num after detach')
	# ceal
	# parser.add_argument('--delta', type=float, default=5 * 1e-5, help='value of delta in ceal sampling')
	#hyper parameters
	parser.add_argument('--train_epochs', '-o', type=int, default=3, help='Number of training epochs')
	parser.add_argument("--local_rank", default=3, type=int)
	
	args = parser.parse_args()
	return args