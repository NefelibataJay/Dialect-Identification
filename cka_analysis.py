from sklearn.metrics.pairwise import cosine_similarity, linear_kernel, rbf_kernel
import matplotlib.pyplot as plt
import numpy as np
import os
import ast
from tqdm import tqdm

def center_kernel(kernel):
    """ 中心化核矩阵 """
    n = kernel.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ kernel @ H

def compute_cka(X, Y, kernel='linear', sigma=None):
    """ 计算CKA 
    Args:
        X: [n_samples, n_features]
        Y: [n_samples, n_features]
        kernel: str, 'linear' or 'rbf'
        sigma: float, rbf kernel parameter
    Returns:
        float, CKA value
    """
    if kernel == 'linear':
        X = X @ X.T
        Y = Y @ Y.T
    elif kernel == 'rbf':
        X = rbf_kernel(X, gamma=1/(2*sigma**2))
        Y = rbf_kernel(Y, gamma=1/(2*sigma**2))
    else:
        raise ValueError('Invalid kernel type')

    X_centered = center_kernel(X)
    Y_centered = center_kernel(Y)

    numerator = np.sum(X_centered * Y_centered)
    denominator = np.sqrt(np.sum(X_centered ** 2) * np.sum(Y_centered ** 2))

    return numerator / denominator



