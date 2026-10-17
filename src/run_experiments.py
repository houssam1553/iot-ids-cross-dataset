"""Step 3 - the experiment.

For every split protocol (random, temporal), seed and model:
  * train on the training split of a SOURCE dataset,
  * pick the decision threshold on the source validation split (max F1),
  * evaluate on the test split of EVERY dataset, including the source itself.

Because each target's test split is identical whether the model was trained on
the target or on another dataset, the "within vs cross" gap is a clean
comparison: only the training network changes.

Usage:
    python -m src.run_experiments                 # 3 seeds, rf + xgb + mlp
    python -m src.run_experiments --quick         # 1 seed, smaller train sets
"""
import argparse
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from .config import DATASETS, LABEL_COL, ATTACK_COL, PROCESSED_DIR, RESULTS_DIR
from .features import shared_features, to_matrix
from .io_utils import load_frame
from .metrics import best_f1_threshold, binary_metrics, per_class_rates
from .models import fit_and_scorer
from .splits import SPLITS


def maybe_subsample(df, n, seed):
    if n and len(df) > n:
        return df.groupby(ATTACK_COL, group_keys=False).sample(frac=n / len(df), random_state=seed)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=PROCESSED_DIR)
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--models", nargs="+", default=["rf", "xgb", "mlp"])
    ap.add_argument("--protocols", nargs="+", default=["random", "temporal"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--max-train", type=int, default=200_000, help="cap on training rows (0 = no cap)")
    ap.add_argument("--dedup", action="store_true", help="drop exact duplicate flows before splitting")
    ap.add_argument("--drop-ttl", action="store_true", help="ablation: also exclude MIN_TTL/MAX_TTL")
    ap.add_argument("--quick", action="store_true", help="1 seed, 50k training rows")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.max_train = args.seeds[:1], 50_000

    dfs = {n: load_frame(args.data_dir / n) for n in args.datasets}
    feats, report = shared_features(dfs, args.drop_ttl)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out_dir / "feature_check.csv", index=False)
    print(f"{len(feats)} shared features used: {feats}")
    if args.dedup:
        for n in dfs:
            before = len(dfs[n])
            dfs[n] = dfs[n].loc[~dfs[n][feats + [LABEL_COL]].duplicated()]
            print(f"dedup {n}: {before:,} -> {len(dfs[n]):,}")

    rows, class_rows = [], []
    t0 = time.time()
    for protocol in args.protocols:
        for seed in args.seeds:
            splits = {n: SPLITS[protocol](df, seed) for n, df in dfs.items()}
            tests = {n: (to_matrix(s[2], feats), s[2][LABEL_COL].to_numpy(), s[2][ATTACK_COL]) for n, s in splits.items()}
            for model in args.models:
                for src, (tr, va, _) in splits.items():
                    tr = maybe_subsample(tr, args.max_train, seed)
                    score = fit_and_scorer(model, to_matrix(tr, feats), tr[LABEL_COL].to_numpy(), seed)
                    thr, _ = best_f1_threshold(va[LABEL_COL].to_numpy(), score(to_matrix(va, feats)))
                    for tgt, (Xt, yt, at) in tests.items():
                        s = score(Xt)
                        m = binary_metrics(yt, s, thr)
                        o_thr, o_f1 = best_f1_threshold(yt, s)
                        key = dict(protocol=protocol, seed=seed, model=model, source=src, target=tgt,
                                   setting="within" if src == tgt else "cross")
                        rows.append({**key, "threshold": thr, **m,
                                     "oracle_threshold": o_thr, "oracle_f1": o_f1,
                                     "n_test": len(yt), "test_attack_share": float(yt.mean())})
                        for r in per_class_rates(at, s, thr).to_dict("records"):
                            class_rows.append({**key, **r})
                    print(f"[{time.time() - t0:6.0f}s] {protocol:8s} seed={seed} {model:3s} trained on {src}")

    res = pd.DataFrame(rows)
    res.to_csv(args.out_dir / "results_all.csv", index=False)
    pd.DataFrame(class_rows).to_csv(args.out_dir / "results_per_class.csv", index=False)
    (args.out_dir / "run_config.json").write_text(json.dumps(
        {k: (str(v) if not isinstance(v, (int, float, bool, list)) else v) for k, v in vars(args).items()}
        | {"features": feats}, indent=2))
    print(f"\nwrote {args.out_dir / 'results_all.csv'} ({len(res)} rows). Next: python -m src.report")


if __name__ == "__main__":
    main()
