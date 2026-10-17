"""Generate small FAKE CSVs with the NetFlow v3 schema, only to test the pipeline.

The numbers these files produce mean nothing. Never report them.

Usage: python tests/make_synthetic.py OUT_DIR [rows_per_dataset]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import V3_FEATURES  # noqa: E402

SPECS = {  # file name, classes, benign share, dataset-specific shift
    "NF-ToN-IoT-v3.csv": (["Benign", "DDoS", "Scanning", "XSS", "Password"], 0.6, 0.0),
    "NF-BoT-IoT-v3.csv": (["Benign", "DDoS", "DoS", "Reconnaissance", "Theft"], 0.2, 1.0),
    "NF-UNSW-NB15-v3.csv": (["Benign", "Exploits", "Fuzzers", "Reconnaissance", "DoS"], 0.9, -0.7),
}


def make(path, classes, benign_share, shift, n, rng):
    probs = [benign_share] + [(1 - benign_share) / (len(classes) - 1)] * (len(classes) - 1)
    attack = rng.choice(classes, size=n, p=probs)
    df = pd.DataFrame(index=range(n))
    numeric = [c for c in V3_FEATURES if not c.startswith("IPV4") and c not in ("FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS")]
    for j, c in enumerate(numeric):
        base = rng.normal(3 + shift, 1, n)
        is_att = attack != "Benign"
        # a weak shared attack signal on some features, plus class-specific offsets
        base += np.where(is_att, 0.8 * ((j % 3) == 0), 0)
        base += np.array([hash((cl, j)) % 5 / 5 for cl in attack]) * is_att
        df[c] = np.round(np.expm1(np.clip(base, 0, 20))).astype("int64")
    t0 = 1_600_000_000_000 + int(shift * 1e9)
    order = np.argsort(attack, kind="stable")          # attacks in time blocks
    ts = np.empty(n, dtype="int64"); ts[order] = t0 + np.sort(rng.integers(0, 3_600_000 * 24, n))
    df["FLOW_START_MILLISECONDS"] = ts
    df["FLOW_END_MILLISECONDS"] = ts + df["FLOW_DURATION_MILLISECONDS"]
    df["IPV4_SRC_ADDR"] = [f"192.168.{1 + (a != 'Benign')}.{rng.integers(1, 250)}" for a in attack]
    df["IPV4_DST_ADDR"] = [f"10.0.0.{rng.integers(1, 250)}" for _ in range(n)]
    df = df[V3_FEATURES]
    df["Label"] = (attack != "Benign").astype(int)
    df["Attack"] = attack
    df.sort_values("FLOW_START_MILLISECONDS").to_csv(path, index=False)


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000
    rng = np.random.default_rng(0)
    for name, (cls, b, s) in SPECS.items():
        make(out / name, cls, b, s, n, rng)
        print("wrote", out / name)
