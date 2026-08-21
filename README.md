# SENTRY V3 — Autoencoder Anomaly Detector

Autoencoder-based network intrusion detector for the SENTRY project.
Trains on normal traffic, flags anything that reconstructs badly.

## Architecture

```
78 → 128 → 96 → 64 → 32 → 16 → 32 → 64 → 96 → 128 → 78
   encoder                bottleneck              decoder
```

Bigger than V2 (deeper encoder/decoder, tighter 16-dim bottleneck).
Input is 78 features matching CIC-IDS2017 flow features.

## Files

| file | purpose |
|---|---|
| `model.py` | The autoencoder + activation hooks for viz |
| `train.py` | Loads CIC-IDS2017, trains on BENIGN only |
| `visualize.py` | Green bowtie diagram + live activation animation |
| `live_capture.py` | Scapy packet sniffer, flow tracker, live scoring |
| `main.py` | Entry point |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download CIC-IDS2017 from Kaggle:
https://www.kaggle.com/datasets/cicdataset/cicids2017

Drop the CSVs into `data/CIC-IDS2017/`.

## Commands

```bash
python main.py train              # trains + saves to checkpoints/
python main.py viz                # static architecture diagram
python main.py live-viz           # animated neuron activations
sudo -E python main.py capture eth0   # live traffic scoring
```

## Tuning notes

- `ANOMALY_THRESHOLD` in `live_capture.py` starts at 0.05. After
  training, run on a validation set and set threshold at ~99th
  percentile of benign reconstruction MSE.
- Live capture uses reduced feature extraction (best-effort mapping
  to training columns). For production parity swap in CICFlowMeter.
- Bottleneck size (16) controls how "compressed" the normal manifold
  is. Too small → underfits benign; too large → memorizes attacks.
