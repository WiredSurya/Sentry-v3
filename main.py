"""
SENTRY V3 entry point.

Commands:
  python main.py train              Train on CIC-IDS2017 (needs data/)
  python main.py viz                Static bowtie diagram
  python main.py live-viz           Animated neuron activations (random data if no ckpt)
  python main.py capture [iface]    Live packet scoring (needs sudo)
"""
from __future__ import annotations
import sys
from pathlib import Path

USAGE = __doc__


def cmd_train():
    from train import train
    train()


def cmd_viz():
    import matplotlib.pyplot as plt
    from model import SentryAutoencoderV3
    from visualize import draw_architecture
    m = SentryAutoencoderV3(input_dim=78)
    draw_architecture(m)
    plt.show()


def cmd_live_viz():
    import torch, joblib
    from model import SentryAutoencoderV3
    from visualize import live_activation_viz

    ckpt = Path("checkpoints/sentry_v3.pt")
    feats = Path("checkpoints/features.pkl")
    if ckpt.exists() and feats.exists():
        cols = joblib.load(feats)
        m = SentryAutoencoderV3(input_dim=len(cols))
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"[live-viz] loaded trained model, input_dim={len(cols)}")
    else:
        print("[live-viz] no checkpoint — using random weights for demo")
        m = SentryAutoencoderV3(input_dim=78)
    m.eval()
    X = torch.rand(30, m.input_dim)
    live_activation_viz(m, X)


def cmd_capture():
    from live_capture import run
    iface = sys.argv[2] if len(sys.argv) > 2 else None
    run(iface)


def main():
    if len(sys.argv) < 2:
        print(USAGE); return
    {
        "train": cmd_train,
        "viz": cmd_viz,
        "live-viz": cmd_live_viz,
        "capture": cmd_capture,
    }.get(sys.argv[1], lambda: print(USAGE))()


if __name__ == "__main__":
    main()
