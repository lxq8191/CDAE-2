from sklearn.model_selection import train_test_split
import numpy as np


def gen_train_test(inputs, ratio=0.3):

    train_rating = np.zeros(
            (inputs.shape[0], inputs.shape[1]),
            dtype=np.float32)

    train_indices = []
    test_indices = []

    for usr in range(inputs.shape[0]):
        non_zero_indices = np.nonzero(inputs[usr])[0]
        non_zero_indices = np.random.permutation(non_zero_indices)
        
        train_idx = []
        test_idx = []
        for idx in range(int((1-ratio)*non_zero_indices.shape[0])):
            train_rating[usr][non_zero_indices[idx]] = 1
            train_idx.append(non_zero_indices[idx])

        for idx in range(idx+1, non_zero_indices.shape[0]):
            test_idx.append(non_zero_indices[idx])

        train_indices.append(train_idx)
        test_indices.append(test_idx)
        
    return train_rating, train_indices, test_indices


def avg_precision(topN, indices):
    '''
    Calculate Average Precision

    -- Args --:
        topN: reconstruct top N recommand
        indices: list of items that user have seen in test set

    -- Return --:
        ap: user's average precision
    '''
    N = len(topN)
    sum_p = 0.
    hit_count = 0

    for i in range(N):
        if topN[i] in indices:
            hit_count += 1
            sum_p += hit_count / (i+1)
    
    try:
        return sum_p / min(N, len(indices))
    except ZeroDivisionError:
        return 100


def get_topN(rec_matrix, train_index, N=5):
    
    topN = []
    recon_rank = rec_matrix[0].argsort()[::-1]

    for rank_idx in recon_rank:
        if len(topN) == N:
            break
        if rank_idx not in train_index:
            topN.append(rank_idx)

    return topN
