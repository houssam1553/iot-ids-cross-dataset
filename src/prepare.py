"""Step 1 - turn the huge raw CSVs into small, class-capped samples.

The raw files are too big for a laptop (NF-ToN-IoT-v3 has ~27.5M flows), so we
read them in chunks and keep a random subset of each attack class:

  pass 1: count flows per class (only the `Attack` column is read)
  pass 2: keep each row of class c with probability min(1, cap_c / count_c)

Row order (and therefore time order) is preserved, which the temporal split needs.

Usage:
    python -m src.prepare                       # default caps
    python -m src.prepare --benign-cap 50000 --attack-cap 10000   # quick run
"""
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .config import (ATTACK_COL, DATASETS, EXPECTED_COUNTS, LABEL_COL, OPTIONAL_DATASETS,
                     PROCESSED_DIR, RAW_DIR, RESULTS_DIR, TIME_COLS, V3_FEATURES)
from .io_utils import find_csv, normalise_attack, save_frame

CHUNK = 1_000_000


def stripped_header(path):
    return [c.strip() for c in pd.read_csv(path, nrows=0).columns]


def count_classes(path) -> pd.Series:
    counts = pd.Series(dtype="int64")
    for chunk in pd.read_csv(path, usecols=lambda c: c.strip() == ATTACK_COL, chunksize=CHUNK):
        chunk.columns = [c.strip() for c in chunk.columns]
        vc = normalise_attack(chunk[ATTACK_COL]).value_counts()
        counts = counts.add(vc, fill_value=0)
    return counts.astype("int64").sort_values(ascending=False)


def sample_file(path, rates: dict, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    kept = []
    for chunk in pd.read_csv(path, chunksize=CHUNK, low_memory=False):
        chunk.columns = [c.strip() for c in chunk.columns]
        chunk[ATTACK_COL] = normalise_attack(chunk[ATTACK_COL])
        p = chunk[ATTACK_COL].map(rates).fillna(1.0).to_numpy()
        kept.append(chunk[rng.random(len(chunk)) < p])
    df = pd.concat(kept, ignore_index=True)
    # float64 -> float32 saves memory; keep timestamps exact (ms since epoch)
    for c in df.select_dtypes("float64").columns:
        if c not in TIME_COLS:
            df[c] = df[c].astype("float32")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--benign-cap", type=int, default=150_000, help="max benign flows kept per dataset")
    ap.add_argument("--attack-cap", type=int, default=30_000, help="max flows kept per attack class")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = ap.parse_args()

    all_ds = {**DATASETS, **OPTIONAL_DATASETS}
    report = {}
    for name in args.datasets:
        path = find_csv(args.raw_dir, all_ds[name])
        if path is None:
            print(f"[skip] {name}: no CSV matching {all_ds[name]} under {args.raw_dir}")
            continue
        print(f"\n=== {name}: {path}")

        header = stripped_header(path)
        missing = [c for c in V3_FEATURES if c not in header]
        extra = [c for c in header if c not in V3_FEATURES + [LABEL_COL, ATTACK_COL]]
        print(f"columns: {len(header)} | missing vs v3 schema: {missing or 'none'} | extra: {extra or 'none'}")

        counts = count_classes(path)
        print("full class counts:\n" + counts.to_string())
        expected = EXPECTED_COUNTS.get(name, {})
        diffs = {k: (int(counts.get(k, 0)), v) for k, v in expected.items() if int(counts.get(k, 0)) != v}
        if diffs:
            print(f"[warn] counts differ from the official UQ page (got, expected): {diffs}")

        rates = {c: min(1.0, (args.benign_cap if c == "Benign" else args.attack_cap) / n)
                 for c, n in counts.items()}
        df = sample_file(path, rates, args.seed)

        mism = int(((df[ATTACK_COL] == "Benign") != (df[LABEL_COL] == 0)).sum())
        if mism:
            print(f"[warn] {mism} rows where Label and Attack disagree")
        out = save_frame(df, args.out_dir / name)
        print(f"kept {len(df):,} rows -> {out}")
        print(df[ATTACK_COL].value_counts().to_string())

        report[name] = {
            "file": str(path.name), "n_columns": len(header),
            "missing_vs_v3_schema": missing, "extra_columns": extra,
            "full_counts": counts.to_dict(), "official_count_mismatches": diffs,
            "sampled_counts": df[ATTACK_COL].value_counts().to_dict(),
            "label_attack_mismatches": mism,
            "caps": {"benign": args.benign_cap, "attack_class": args.attack_cap, "seed": args.seed},
        }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "data_report.json").write_text(json.dumps(report, indent=2, default=int))
    print(f"\nwrote {args.results_dir / 'data_report.json'}")


if __name__ == "__main__":
    main()
