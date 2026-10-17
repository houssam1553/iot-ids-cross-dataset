"""Step 2 - leakage audit, run BEFORE trusting any score.

For each dataset it reports:
  1. exact duplicates: share of flows whose feature vector appears more than
     once, and share of those duplicates that carry BOTH labels;
  2. train/test overlap: share of test flows whose exact feature vector is
     also in the training set, under the random and the temporal split;
  3. single-feature shortcuts: ROC-AUC of a depth-3 tree that sees ONE column.
     Identifiers and timestamps are included on purpose, to show why they are
     excluded. A modelled feature with AUC ~1.0 deserves a closer look.

Usage: python -m src.audit
"""
import argparse
from pathlib import Path
import ipaddress

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from .config import (DATASETS, IDENTIFIER_COLS, LABEL_COL, PROCESSED_DIR, RESULTS_DIR,
                     TIME_COLS, excluded_columns)
from .features import shared_features, to_matrix
from .io_utils import load_frame, md_table
from .splits import random_split, temporal_split


def row_hashes(df, feats):
    return pd.util.hash_pandas_object(pd.DataFrame(to_matrix(df, feats)), index=False).to_numpy()


def ip_to_int(s: pd.Series) -> pd.Series:
    def conv(v):
        try:
            return int(ipaddress.ip_address(str(v).strip())) % (2**53)
        except ValueError:
            return 0
    return s.map(conv).astype("float64")


def audit_dataset(name, df, feats, seed=0):
    out = {"dataset": name, "rows": len(df)}
    h = pd.Series(row_hashes(df, feats))
    dup = h.duplicated(keep=False)
    out["dup_rows_pct"] = 100 * dup.mean()
    lab = df[LABEL_COL].to_numpy()
    conflict = pd.DataFrame({"h": h, "y": lab}).groupby("h")["y"].nunique()
    out["conflicting_dup_pct"] = 100 * h.isin(conflict[conflict > 1].index).mean()

    for split_name, split in (("random", random_split), ("temporal", temporal_split)):
        tr, _, te = split(df, seed)
        seen = set(row_hashes(tr, feats))
        out[f"test_in_train_pct_{split_name}"] = 100 * np.mean([x in seen for x in row_hashes(te, feats)])

    # single-feature shortcut check
    tr, _, te = random_split(df, seed)
    cand = [c for c in df.columns if c not in ("Label", "Attack", "Dataset")]
    rows = []
    for c in cand:
        conv = ip_to_int if df[c].dtype == object else (lambda s: s.astype("float64"))
        try:
            xtr, xte = conv(tr[c]).to_numpy()[:, None], conv(te[c]).to_numpy()[:, None]
        except (TypeError, ValueError):
            continue
        xtr = np.nan_to_num(xtr, posinf=0, neginf=0); xte = np.nan_to_num(xte, posinf=0, neginf=0)
        tree = DecisionTreeClassifier(max_depth=3, random_state=seed).fit(xtr, tr[LABEL_COL])
        try:
            auc = roc_auc_score(te[LABEL_COL], tree.predict_proba(xte)[:, 1])
        except ValueError:
            auc = float("nan")
        role = ("identifier" if c in IDENTIFIER_COLS else "timestamp" if c in TIME_COLS
                else "used" if c in feats else "not used")
        rows.append({"dataset": name, "feature": c, "role": role, "single_feature_auc": auc})
    return out, pd.DataFrame(rows).sort_values("single_feature_auc", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=PROCESSED_DIR)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--drop-ttl", action="store_true")
    args = ap.parse_args()

    dfs = {}
    for n in args.datasets:
        try:
            dfs[n] = load_frame(args.data_dir / n)
        except FileNotFoundError as e:
            print(f"[skip] {e}")
    feats, report = shared_features(dfs, args.drop_ttl)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(RESULTS_DIR / "feature_check.csv", index=False)
    print(f"{len(feats)} shared model features (details: results/feature_check.csv)")

    summary, shortcuts = [], []
    for n, df in dfs.items():
        s, sc = audit_dataset(n, df, feats)
        summary.append(s); shortcuts.append(sc)
        print(f"\n{n}: " + ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items()))
        print(sc.head(8).to_string(index=False))
    summary = pd.DataFrame(summary); shortcuts = pd.concat(shortcuts)
    summary.to_csv(RESULTS_DIR / "audit_summary.csv", index=False)
    shortcuts.to_csv(RESULTS_DIR / "audit_single_feature_auc.csv", index=False)

    md = ["# Leakage audit", "",
          "## Duplicates and train/test overlap (% of flows)", "",
          md_table(summary.round(2), '{:.2f}'), "",
          "## Strongest single features (depth-3 tree, random split, ROC-AUC)", ""]
    for n in dfs:
        md += [f"### {n}", "", md_table(shortcuts[shortcuts.dataset == n].head(10)
               .drop(columns="dataset"), '{:.4f}'), ""]
    (RESULTS_DIR / "audit.md").write_text("\n".join(md))
    print("\nwrote results/audit.md")


if __name__ == "__main__":
    main()
