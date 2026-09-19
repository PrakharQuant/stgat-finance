import numpy as np

def correlation_adj(returns_window: np.ndarray, threshold: float = 0.4):
    """
    returns_window: [N, L]
    Paper: A'_ij = sigmoid(r_ij) if r_ij > alpha else 0.
    We use |Pearson| so crash co-movement is an edge too.
    """
    C = np.corrcoef(returns_window)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 1.0)
    A = np.abs(C)
    A = np.where(A >= threshold, A, 0.0)
    return A.astype(np.float32)
