import numpy as np
import torch
from torch.utils.data import Dataset
from .stl import decompose_panel
from .graph import correlation_adj

class IndexVolDataset(Dataset):
    def __init__(self, close, returns, vol, cfg, start, end):
        self.cfg = cfg
        c = close.iloc[start:end]
        r = returns.iloc[start:end]
        v = vol.iloc[start:end]
        self.X = c.values.T.astype(np.float32)          # [N, T]
        self.R = r.values.T.astype(np.float32)
        self.Y = v.values.T.astype(np.float32)
        # z-score prices per node on this split only
        mu = self.X.mean(axis=1, keepdims=True)
        sd = self.X.std(axis=1, keepdims=True) + 1e-8
        self.Xn = (self.X - mu) / sd
        self.trend, self.seas, self.resid = decompose_panel(self.Xn, cfg["stl_period"])
        self.w = cfg["window"]
        self.h = cfg["horizon"]
        self.step = cfg["step"]
        self.corr_lb = cfg["corr_lookback"]
        self.idx = list(range(self.w, self.Xn.shape[1] - self.h, self.step))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        t = self.idx[k]
        sl = slice(t - self.w, t)
        # component windows [N, W]
        trend = self.trend[:, sl]
        seas = self.seas[:, sl]
        resid = self.resid[:, sl]
        lb0 = max(0, t - self.corr_lb)
        A = correlation_adj(self.R[:, lb0:t], self.cfg["corr_threshold"])
        y = self.Y[:, t : t + self.h]                   # [N, H]
        return {
            "trend": torch.from_numpy(trend),
            "seas": torch.from_numpy(seas),
            "resid": torch.from_numpy(resid),
            "A": torch.from_numpy(A),
            "y": torch.from_numpy(y),
        }
