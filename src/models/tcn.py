import torch.nn as nn
from torch.nn.utils import weight_norm

class CausalConv1d(nn.Module):
    def __init__(self, cin, cout, k, dilation):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = weight_norm(nn.Conv1d(cin, cout, k, dilation=dilation))

    def forward(self, x):
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)

class TCNBlock(nn.Module):
    def __init__(self, channels, k=3, levels=4, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(levels):
            layers += [
                CausalConv1d(channels, channels, k, dilation=2 ** i),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        self.net = nn.Sequential(*layers)
        self.res = nn.Identity()

    def forward(self, x):
        # x: [B*N, C, W]
        return self.net(x) + self.res(x)
