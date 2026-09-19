import torch
import torch.nn as nn
from .gat import MultiHeadGAT
from .tcn import TCNBlock

class ComponentBranch(nn.Module):
    def __init__(self, hidden, heads, tcn_channels, tcn_levels, k, dropout):
        super().__init__()
        self.in_proj = nn.Linear(1, hidden)
        self.gat = MultiHeadGAT(hidden, hidden, heads, dropout)
        self.to_tcn = nn.Linear(hidden, tcn_channels)
        self.tcn = TCNBlock(tcn_channels, k, tcn_levels, dropout)

    def forward(self, comp, A):
        # comp: [B, N, W]
        B, N, W = comp.shape
        x = self.in_proj(comp.unsqueeze(-1))             # [B, N, W, H]
        spatial = []
        for t in range(W):
            spatial.append(self.gat(x[:, :, t, :], A))
        spat = torch.stack(spatial, dim=2)               # [B, N, W, H]
        z = self.to_tcn(spat)                            # [B, N, W, C]
        z = z.reshape(B * N, W, -1).transpose(1, 2)      # [B*N, C, W]
        z = self.tcn(z).transpose(1, 2)                  # [B*N, W, C]
        return z.reshape(B, N, W, -1)

class STGAT(nn.Module):
    def __init__(self, cfg, n_nodes):
        super().__init__()
        H, C = cfg["hidden"], cfg["tcn_channels"]
        kw = dict(hidden=H, heads=cfg["heads"], tcn_channels=C,
                  tcn_levels=cfg["tcn_levels"], k=cfg["kernel"],
                  dropout=cfg["dropout"])
        self.trend = ComponentBranch(**kw)
        self.seas = ComponentBranch(**kw)
        self.resid_st = ComponentBranch(**kw)
        self.resid_mlp = nn.Sequential(
            nn.Linear(cfg["window"], H),
            nn.ReLU(),
            nn.Linear(H, C),
        )
        fuse_in = 3 * C + C
        self.fuse_in = nn.Linear(fuse_in, C)
        self.fuse_tcn = TCNBlock(C, cfg["kernel"], cfg["tcn_levels"], cfg["dropout"])
        self.head = nn.Sequential(
            nn.Linear(C, C),
            nn.ReLU(),
            nn.Linear(C, cfg["horizon"]),
        )
        self.n_nodes = n_nodes

    def forward(self, batch):
        A = batch["A"]
        t = self.trend(batch["trend"], A)
        s = self.seas(batch["seas"], A)
        r = self.resid_st(batch["resid"], A)
        mlp = self.resid_mlp(batch["resid"])             # [B, N, C]
        mlp = mlp.unsqueeze(2).expand(-1, -1, t.size(2), -1)
        cat = torch.cat([t, s, r, mlp], dim=-1)
        B, N, W, F = cat.shape
        z = self.fuse_in(cat).reshape(B * N, W, -1).transpose(1, 2)
        z = self.fuse_tcn(z).transpose(1, 2)             # [B*N, W, C]
        last = z[:, -1, :]
        y = self.head(last).reshape(B, N, -1)
        return y
