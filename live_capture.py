"""
Live network traffic scoring.

WHY this is non-trivial:
  CIC-IDS2017 features are FLOW-level (aggregated over multiple packets
  in the same 5-tuple), not per-packet. So we can't just score every
  packet independently. We bucket packets into flows keyed by
  (src_ip, src_port, dst_ip, dst_port, proto), and score a flow when
  it closes (FIN/RST) or times out.

  Our extracted features are a REDUCED, best-effort subset of what
  CICFlowMeter produces. `align_features` maps our dict onto the exact
  column order the model was trained on, filling gaps with 0. This
  isn't perfect parity with the training data — for production you'd
  wire in real CICFlowMeter output — but it's enough to see real
  traffic light up the visualizer and produce meaningful scores.

Needs sudo/root to sniff: `sudo -E python main.py capture eth0`
"""
from __future__ import annotations
import time
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import joblib

from model import SentryAutoencoderV3

try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP
except ImportError:
    raise SystemExit("scapy missing.  pip install scapy")

MODEL_PATH = Path("checkpoints/sentry_v3.pt")
SCALER_PATH = Path("checkpoints/scaler.pkl")
FEATURES_PATH = Path("checkpoints/features.pkl")

FLOW_TIMEOUT_S = 30.0
ANOMALY_THRESHOLD = 0.05  # tune after training — check score dist first


class FlowTracker:
    def __init__(self):
        self.flows = defaultdict(lambda: {"pkts": [], "start": None, "last": None})

    def add(self, pkt):
        if IP not in pkt:
            return None
        ip = pkt[IP]
        if TCP in pkt:
            proto, sport, dport, flags = "TCP", pkt[TCP].sport, pkt[TCP].dport, int(pkt[TCP].flags)
        elif UDP in pkt:
            proto, sport, dport, flags = "UDP", pkt[UDP].sport, pkt[UDP].dport, 0
        elif ICMP in pkt:
            proto, sport, dport, flags = "ICMP", 0, 0, 0
        else:
            proto, sport, dport, flags = "OTHER", 0, 0, 0

        key = (ip.src, sport, ip.dst, dport, proto)
        now = time.time()
        f = self.flows[key]
        if f["start"] is None:
            f["start"] = now
        f["last"] = now
        f["pkts"].append({"len": len(pkt), "t": now, "flags": flags})

        # Close on FIN or RST
        if proto == "TCP" and (flags & 0x01 or flags & 0x04):
            return self._close(key)
        return None

    def _close(self, key):
        f = self.flows.pop(key, None)
        if not f or len(f["pkts"]) < 2:
            return None
        return {"key": key, "feats": self._featurize(key, f)}

    def sweep(self):
        now = time.time()
        out = []
        for k in list(self.flows.keys()):
            if now - self.flows[k]["last"] > FLOW_TIMEOUT_S:
                r = self._close(k)
                if r:
                    out.append(r)
        return out

    @staticmethod
    def _featurize(key, f):
        pkts = f["pkts"]
        lens = np.array([p["len"] for p in pkts], dtype=np.float64)
        times = np.array([p["t"] for p in pkts])
        iats = np.diff(times) if len(times) > 1 else np.array([0.0])
        dur = max(f["last"] - f["start"], 1e-6)
        return {
            "flow_duration": dur,
            "total_fwd_packets": len(pkts),
            "total_length_of_fwd_packets": float(lens.sum()),
            "fwd_packet_length_mean": float(lens.mean()),
            "fwd_packet_length_std": float(lens.std()),
            "fwd_packet_length_min": float(lens.min()),
            "fwd_packet_length_max": float(lens.max()),
            "flow_iat_mean": float(iats.mean()),
            "flow_iat_std": float(iats.std()),
            "flow_iat_min": float(iats.min()),
            "flow_iat_max": float(iats.max()),
            "flow_bytes_per_s": float(lens.sum() / dur),
            "flow_packets_per_s": float(len(pkts) / dur),
            "syn_flag_count": sum(1 for p in pkts if p["flags"] & 0x02),
            "ack_flag_count": sum(1 for p in pkts if p["flags"] & 0x10),
            "fin_flag_count": sum(1 for p in pkts if p["flags"] & 0x01),
            "rst_flag_count": sum(1 for p in pkts if p["flags"] & 0x04),
            "psh_flag_count": sum(1 for p in pkts if p["flags"] & 0x08),
            "urg_flag_count": sum(1 for p in pkts if p["flags"] & 0x20),
            "destination_port": key[3],
            "source_port": key[1],
        }


def align_features(feats: dict, target_cols: list) -> np.ndarray:
    """
    Substring-match our extracted keys against the training columns.
    Missing training columns get 0.0. Good enough to smoke-test live
    capture; swap in CICFlowMeter for real parity.
    """
    vec = []
    keys_lower = {k.lower(): v for k, v in feats.items()}
    for col in target_cols:
        cl = col.strip().lower().replace(" ", "_").replace("/", "_per_")
        match = keys_lower.get(cl)
        if match is None:
            for k, v in keys_lower.items():
                if k in cl or cl in k:
                    match = v
                    break
        vec.append(float(match) if match is not None else 0.0)
    return np.array(vec, dtype=np.float32)


def run(iface: str = None):
    scaler = joblib.load(SCALER_PATH)
    cols = joblib.load(FEATURES_PATH)

    model = SentryAutoencoderV3(input_dim=len(cols))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    tracker = FlowTracker()
    print(f"[capture] iface={iface or 'default'}  threshold={ANOMALY_THRESHOLD}")
    print(f"[capture] input_dim={len(cols)}  (Ctrl-C to stop)")

    def handle(pkt):
        closed = tracker.add(pkt)
        for flow in filter(None, [closed] + tracker.sweep()):
            vec = align_features(flow["feats"], cols)
            X = scaler.transform(vec.reshape(1, -1)).astype(np.float32)
            with torch.no_grad():
                score = float(model.anomaly_score(torch.from_numpy(X)).item())
            tag = "🚨 ANOMALY" if score > ANOMALY_THRESHOLD else "  normal "
            k = flow["key"]
            print(f"  [{tag}] score={score:.5f}  {k[0]}:{k[1]} -> {k[2]}:{k[3]} ({k[4]})")

    sniff(iface=iface, prn=handle, store=False)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
