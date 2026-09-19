import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data.fetch import load_cfg, fetch_prices, log_returns, realized_vol
from src.data.dataset import IndexVolDataset
from src.models.stgat import STGAT

def split_idx(n, val_ratio, test_ratio):
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test
    return (0, n_train), (n_train, n_train + n_val), (n_train + n_val, n)

def collate(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}

def main():
    cfg = load_cfg()
    torch.manual_seed(cfg["seed"])
    close = fetch_prices(cfg)
    rets = log_returns(close)
    vol = realized_vol(rets)
    close, rets, vol = close.loc[rets.index], rets, vol.loc[rets.index]
    n = len(close)
    tr, va, te = split_idx(n, cfg["val_ratio"], cfg["test_ratio"])
    ds_tr = IndexVolDataset(close, rets, vol, cfg, *tr)
    ds_va = IndexVolDataset(close, rets, vol, cfg, *va)
    loader = DataLoader(ds_tr, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate)
    vloader = DataLoader(ds_va, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate)

    device = torch.device(cfg["device"])
    model = STGAT(cfg, n_nodes=close.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    best, patience, wait = 1e9, 12, 0
    Path("checkpoints").mkdir(exist_ok=True)

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        tl = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model(batch)
            loss = torch.nn.functional.mse_loss(pred, batch["y"])
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item()
        model.eval()
        vl = 0
        with torch.no_grad():
            for batch in vloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                vl += torch.nn.functional.mse_loss(model(batch), batch["y"]).item()
        vl /= max(len(vloader), 1)
        print(f"epoch {epoch:03d}  train {tl/len(loader):.5f}  val {vl:.5f}")
        if vl < best:
            best, wait = vl, 0
            torch.save({"model": model.state_dict(), "cfg": cfg,
                        "nodes": list(close.columns)}, "checkpoints/stgat.pt")
        else:
            wait += 1
            if wait >= patience:
                print("early stop"); break

if __name__ == "__main__":
    main()
