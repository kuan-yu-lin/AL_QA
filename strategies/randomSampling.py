import numpy as np

def random_sampling(labeled_idxs, n):
    print('Random querying starts.')
    return np.random.choice(np.where(labeled_idxs==0)[0], n, replace=False)