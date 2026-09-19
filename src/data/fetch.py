from pathlib import Path
import pandas as pd
import yfinance as yf
import yaml

def load_cfg(path="configs/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def fetch_prices(cfg) -> pd.DataFrame:
    tickers = list(cfg["tickers"].values())
    names = list(cfg["tickers"].keys())
    raw = yf.download(tickers, start=cfg["start"], auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close = close.reindex(columns=tickers)
    close.columns = names
    # calendar-day align across timezones; do not peek into the future
    close = close.sort_index().ffill().dropna(how="any")
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    close.to_csv("data/raw/closes.csv")
    return close

def log_returns(close: pd.DataFrame) -> pd.DataFrame:
    return (close / close.shift(1)).apply(lambda s: s.map(lambda x: 0 if x <= 0 else __import__("math").log(x))).dropna()

def realized_vol(returns: pd.DataFrame, window: int = 1) -> pd.DataFrame:
    # next-step target: absolute log-return as a simple vol proxy
    return returns.abs()
