import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadGAT(nn.Module):
    """Per-timestep GAT over N market nodes, with residual + concat heads."""

    def __init__(self, in_dim, out_dim, heads=4, dropout=0.2):
        super().__init__()
        self.heads = heads
        self.W = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False) for _ in range(heads)])
        self.a = nn.ModuleList([nn.Linear(2 * out_dim, 1, bias=False) for _ in range(heads)])
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(heads * out_dim, out_dim)
        self.res = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, A):
        # x: [B, N, F], A: [B, N, N] weighted adjacency
        outs = []
        for h in range(self.heads):
            Wh = self.W[h](x)                            # [B, N, D]
            Bi, Bj = Wh.unsqueeze(2), Wh.unsqueeze(1)    # [B,N,1,D], [B,1,N,D]
            cat = torch.cat([Bi.expand(-1, -1, x.size(1), -1),
                             Bj.expand(-1, x.size(1), -1, -1)], dim=-1)
            e = self.leaky(self.a[h](cat).squeeze(-1))   # [B, N, N]
            e = e.masked_fill(A <= 0, -1e9)
            alpha = self.dropout(F.softmax(e, dim=-1))
            h_i = torch.bmm(alpha, Wh)
            outs.append(h_i)
        h = self.proj(torch.cat(outs, dim=-1))
        return F.elu(h + self.res(x))
