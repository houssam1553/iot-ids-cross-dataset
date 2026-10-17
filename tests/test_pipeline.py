"""End-to-end smoke test on FAKE data: `pytest -q` (about 1 minute)."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import excluded_columns  # noqa: E402
from src.metrics import binary_metrics  # noqa: E402
from src.splits import temporal_split  # noqa: E402


def run(*args, cwd=ROOT):
    subprocess.run([sys.executable, *args], cwd=cwd, check=True)


def test_metrics_basic():
    y = np.array([0, 0, 1, 1]); s = np.array([0.1, 0.6, 0.4, 0.9])
    m = binary_metrics(y, s, 0.5)
    assert m["fpr"] == 0.5 and m["detection_rate"] == 0.5 and abs(m["f1"] - 0.5) < 1e-9


def test_temporal_split_is_ordered():
    df = pd.DataFrame({"Attack": ["Benign"] * 10 + ["DoS"] * 10,
                       "FLOW_START_MILLISECONDS": list(range(20))[::-1]})
    tr, va, te = temporal_split(df)
    for cls in ("Benign", "DoS"):
        assert tr[tr.Attack == cls].FLOW_START_MILLISECONDS.max() < te[te.Attack == cls].FLOW_START_MILLISECONDS.min()


def test_identifiers_never_used():
    ex = excluded_columns()
    for c in ["IPV4_SRC_ADDR", "L4_DST_PORT", "FLOW_START_MILLISECONDS", "Label", "Attack"]:
        assert c in ex


def test_end_to_end(tmp_path):
    raw, proc, res = tmp_path / "raw", tmp_path / "proc", tmp_path / "res"
    run("tests/make_synthetic.py", str(raw), "4000")
    run("-m", "src.prepare", "--raw-dir", str(raw), "--out-dir", str(proc), "--results-dir", str(res),
        "--benign-cap", "1500", "--attack-cap", "400")
    run("-m", "src.run_experiments", "--quick", "--models", "rf", "xgb", "--protocols", "temporal",
        "--data-dir", str(proc), "--out-dir", str(res))
    run("-m", "src.report", "--results-dir", str(res), "--readme", str(tmp_path / "none.md"))
    out = pd.read_csv(res / "results_all.csv")
    assert set(out.setting) == {"within", "cross"}
    assert out[["f1", "fpr", "detection_rate"]].notna().all().all()
    assert (res / "key_numbers.json").exists()
