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

    for i in range(N):
        count = 0
        if topN[i] in indices:
            for idx in range(i+1):
                count += 1 if topN[idx] in indices else 0
            sum_p += count / (i+1)
    
    return sum_p / min(N, len(indices))
