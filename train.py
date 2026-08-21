"""
Train SENTRY V3 on CIC-IDS2017 benign traffic.

WHY train on benign only:
  Anomaly autoencoders learn the manifold of "normal". If we mixed attacks
  in, the model would happily learn to reconstruct attacks too and lose
  discriminative power. So we filter Label == "BENIGN" before training.

Data expected at ./data/CIC-IDS2017/*.csv
Get it from Kaggle: https://www.kaggle.com/datasets/cicdataset/cicids2017
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import joblib

from model import SentryAutoencoderV3

DATA_DIR = Path("data/CIC-IDS2017")
CKPT_DIR = Path("checkpoints")
MODEL_PATH = CKPT_DIR / "sentry_v3.pt"
SCALER_PATH = CKPT_DIR / "scaler.pkl"
FEATURES_PATH = CKPT_DIR / "features.pkl"

BATCH_SIZE = 512
EPOCHS = 50
LR = 1e-3


def load_benign() -> pd.DataFrame:
    csvs = sorted(DATA_DIR.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No CSVs in {DATA_DIR}. Download CIC-IDS2017 and drop them there."
        )
    print(f"[data] found {len(csvs)} csv(s)")
    df = pd.concat((pd.read_csv(f, low_memory=False) for f in csvs), ignore_index=True)
    df.columns = df.columns.str.strip()

    label_col = "Label"
    if label_col not in df.columns:
        raise KeyError(f"'Label' column missing. Got: {df.columns.tolist()[:5]}...")

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    benign = df[df[label_col].astype(str).str.upper() == "BENIGN"].drop(columns=[label_col])
    benign = benign.select_dtypes(include=[np.number])
    print(f"[data] {len(benign):,} benign rows, {benign.shape[1]} numeric features")
    return benign


def train():
    CKPT_DIR.mkdir(exist_ok=True)
    df = load_benign()

    # MinMax to [0,1] so Sigmoid output layer matches the target range
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df.values.astype(np.float32))
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(df.columns.tolist(), FEATURES_PATH)

    tensor = torch.from_numpy(X)
    loader = DataLoader(
        TensorDataset(tensor, tensor),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentryAutoencoderV3(input_dim=X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    print(f"[train] device={device}  epochs={EPOCHS}  input_dim={X.shape[1]}")
    for ep in range(1, EPOCHS + 1):
        model.train()
        running, n = 0.0, 0
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            recon, _ = model(xb)
            loss = loss_fn(recon, xb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
            n += xb.size(0)
        print(f"  ep {ep:03d}/{EPOCHS}  loss={running/n:.6f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[train] saved -> {MODEL_PATH}")


if __name__ == "__main__":
    train()
