import numpy as np
from statsmodels.tsa.seasonal import STL

def stl_decompose(series: np.ndarray, period: int = 5):
    """Return trend, seasonal, residual with same length as series."""
    res = STL(series, period=period, robust=True).fit()
    return res.trend.astype(np.float32), res.seasonal.astype(np.float32), res.resid.astype(np.float32)

def decompose_panel(X: np.ndarray, period: int = 5):
    """X: [N, T] → three arrays of shape [N, T]."""
    N, T = X.shape
    trend = np.zeros_like(X, dtype=np.float32)
    seas = np.zeros_like(X, dtype=np.float32)
    resid = np.zeros_like(X, dtype=np.float32)
    for i in range(N):
        trend[i], seas[i], resid[i] = stl_decompose(X[i], period)
    return trend, seas, resid
