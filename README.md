# STGAT-Finance

**Spatial–Temporal Graph Attention Network for cross-market volatility forecasting**

Architecture: **Decompose → Spatial → Temporal → Fuse**

STGAT treats raw index prices as noisy. It first splits each series with STL, then learns **who** matters (residual multi-head GAT over a correlation graph) and **when** it matters (residual TCN), and fuses the three STL streams into a next-step volatility forecast.

Inspired by Feng, Jiang, Liang, and Xia, *STGAT: Spatial–Temporal Graph Attention Neural Network for Stock Prediction*, Applied Sciences, 2025.
DOI: https://doi.org/10.3390/app15084315

This repository is an independent adaptation:

- nodes = global indices and commodities, not CSI 500 / S&P 500 constituents
- target = next-day realized volatility, not next-day price
- data = Yahoo Finance via `yfinance` (no API key)

---

## Why STGAT

| Model | Graph (WHO) | Long memory (WHEN) | Non-stationarity |
| --- | --- | --- | --- |
| GARCH | no | limited | no decomposition |
| LSTM | no | yes, sequential | mixed trend + noise |
| GAT alone | yes | no | mixed |
| **STGAT** | residual multi-head GAT | dilated causal TCN | STL + residual MLP |

One-line summary:

> STGAT disentangles each series into interpretable STL components to reduce non-stationarity, applies a residual multi-head GAT for time-varying cross-market dependence, a residual TCN for long-range temporal motifs, and a fusion TCN to combine multi-scale features for volatility clustering.

---

## Architecture
