import numpy as np

import arguments

args_input = arguments.get_args()
NUM_QUERY = args_input.batch

def query(labeled_idxs):
    return np.random.choice(np.where(labeled_idxs==0)[0], NUM_QUERY, replace=False)