"""Step 4 - turn results/*.csv into tables, a figure, key numbers and README text.

Writes:
  results/results_tables.md     all tables
  results/key_numbers.json      the numbers behind the headline finding
  results/cross_dataset_f1.png  heatmaps (temporal protocol)
and replaces the text between the RESULTS / FINDING markers in README.md.

Usage: python -m src.report
"""
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .config import RESULTS_DIR, ROOT
from .io_utils import md_table
from .models import MODEL_NAMES

DS_NAMES = {"ton_iot": "NF-ToN-IoT-v3", "bot_iot": "NF-BoT-IoT-v3",
            "unsw_nb15": "NF-UNSW-NB15-v3", "cicids2018": "NF-CICIDS2018-v3"}
METRICS = ["f1", "detection_rate", "fpr", "roc_auc", "oracle_f1", "mcc"]


def ms(x: pd.Series, pct=False) -> str:
    m, s = x.mean(), x.std(ddof=0)
    if pct:
        return f"{100 * m:.1f}% ± {100 * s:.1f}" if len(x) > 1 else f"{100 * m:.1f}%"
    return f"{m:.3f} ± {s:.3f}" if len(x) > 1 else f"{m:.3f}"


def headline(res):
    """Average over dataset pairs inside each seed, then mean ± std over seeds."""
    per_seed = res.groupby(["model", "protocol", "setting", "seed"])[METRICS].mean().reset_index()
    rows = []
    order = [("random", "within"), ("temporal", "within"), ("temporal", "cross"), ("random", "cross")]
    for model in per_seed.model.unique():
        for prot, sett in order:
            g = per_seed[(per_seed.model == model) & (per_seed.protocol == prot) & (per_seed.setting == sett)]
            if g.empty:
                continue
            label = {"within": "same dataset", "cross": "other dataset"}[sett]
            rows.append({"Model": MODEL_NAMES.get(model, model), "Split": prot, "Tested on": label,
                         "F1": ms(g.f1), "Detection rate": ms(g.detection_rate, True),
                         "FPR": ms(g.fpr, True), "ROC-AUC": ms(g.roc_auc),
                         "Oracle F1*": ms(g.oracle_f1)})
    return pd.DataFrame(rows)


def matrix(res, model, protocol="temporal"):
    g = res[(res.model == model) & (res.protocol == protocol)]
    g = g.groupby(["source", "target"])[["f1", "fpr"]].mean().reset_index()
    names = list(dict.fromkeys(g.source))
    tab = pd.DataFrame(index=[DS_NAMES.get(n, n) for n in names],
                       columns=[DS_NAMES.get(n, n) for n in names], dtype=object)
    f1 = pd.DataFrame(index=names, columns=names, dtype=float)
    for r in g.itertuples():
        tab.loc[DS_NAMES.get(r.source, r.source), DS_NAMES.get(r.target, r.target)] = f"{r.f1:.2f} ({100 * r.fpr:.1f}%)"
        f1.loc[r.source, r.target] = r.f1
    tab.insert(0, "train \\ test", tab.index)
    return tab, f1


def per_class_table(cls, protocol="temporal"):
    c = cls[cls.protocol == protocol]
    c = c.assign(setting=np.where(c.source == c.target, "within", "cross"))
    g = c.groupby(["target", "attack_class", "setting"])["flagged_rate"].mean().unstack("setting")
    n = c.groupby(["target", "attack_class"])["n"].first()
    g = g.join(n).reset_index()
    g["drop"] = g["within"] - g["cross"]
    g["target"] = g["target"].map(lambda t: DS_NAMES.get(t, t))
    benign = g.attack_class == "Benign"
    g.loc[benign, "attack_class"] = "Benign (false-alarm rate)"
    g = g.sort_values(["target", "drop"], ascending=[True, False])
    out = g.rename(columns={"target": "Test set", "attack_class": "Class", "n": "Test flows",
                            "within": "Flagged, trained same", "cross": "Flagged, trained other",
                            "drop": "Difference"})
    for col in ["Flagged, trained same", "Flagged, trained other", "Difference"]:
        out[col] = out[col].map(lambda v: f"{100 * v:.1f}%")
    return out[["Test set", "Class", "Test flows", "Flagged, trained same", "Flagged, trained other", "Difference"]], g


def key_numbers(res, cls_raw):
    t = res[res.protocol == "temporal"]
    w, c = t[t.setting == "within"], t[t.setting == "cross"]
    rw = res[(res.protocol == "random") & (res.setting == "within")]
    pair = c.groupby(["source", "target"]).f1.mean()
    kn = {
        "n_models": int(res.model.nunique()), "n_seeds": int(res.seed.nunique()),
        "n_cross_pairs": int(len(pair)),
        "random_within_f1": rw.f1.mean(), "temporal_within_f1": w.f1.mean(), "cross_f1": c.f1.mean(),
        "temporal_within_fpr": w.fpr.mean(), "cross_fpr": c.fpr.mean(),
        "temporal_within_dr": w.detection_rate.mean(), "cross_dr": c.detection_rate.mean(),
        "temporal_within_auc": w.roc_auc.mean(), "cross_auc": c.roc_auc.mean(),
        "cross_oracle_f1": c.oracle_f1.mean(),
        "worst_pair": f"{DS_NAMES.get(pair.idxmin()[0])} -> {DS_NAMES.get(pair.idxmin()[1])}" if len(pair) else None,
        "worst_pair_f1": float(pair.min()) if len(pair) else None,
        "best_pair": f"{DS_NAMES.get(pair.idxmax()[0])} -> {DS_NAMES.get(pair.idxmax()[1])}" if len(pair) else None,
        "best_pair_f1": float(pair.max()) if len(pair) else None,
    }
    gap = kn["temporal_within_f1"] - kn["cross_f1"]
    kn["share_of_gap_recovered_by_oracle_threshold"] = (
        (kn["cross_oracle_f1"] - kn["cross_f1"]) / gap if gap > 1e-9 else None)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in kn.items()}


def draft_finding(kn):
    s = (f"Across {kn['n_models']} standard models (Random Forest, XGBoost, MLP), {kn['n_cross_pairs']} "
         f"train/test dataset pairs and {kn['n_seeds']} seed(s), F1 was {kn['random_within_f1']:.2f} with the usual "
         f"random split, {kn['temporal_within_f1']:.2f} with a time-ordered split on the same dataset, and "
         f"{kn['cross_f1']:.2f} when the model was tested on a different dataset with the same 53-feature NetFlow format; "
         f"the false-positive rate went from {100 * kn['temporal_within_fpr']:.1f}% to {100 * kn['cross_fpr']:.1f}%.")
    r = kn.get("share_of_gap_recovered_by_oracle_threshold")
    if r is not None and r >= 0.5:
        s += (f" Re-tuning only the decision threshold on the new network would recover {100 * r:.0f}% of the F1 gap "
              f"(ROC-AUC {kn['temporal_within_auc']:.2f} -> {kn['cross_auc']:.2f}), so much of the failure is miscalibration "
              f"rather than a total loss of signal.")
    elif r is not None:
        s += (f" Even the best possible threshold on the new network recovers only {100 * r:.0f}% of the gap "
              f"(ROC-AUC {kn['temporal_within_auc']:.2f} -> {kn['cross_auc']:.2f}): what the models learn does not transfer.")
    return s


def replace_block(text, tag, new):
    start, end = f"<!-- {tag}:START -->", f"<!-- {tag}:END -->"
    if start not in text:
        return text
    a, b = text.index(start) + len(start), text.index(end)
    return text[:a] + "\n" + new.strip() + "\n" + text[b:]


def heatmaps(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = list(res.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.2), squeeze=False)
    for ax, m in zip(axes[0], models):
        _, f1 = matrix(res, m)
        im = ax.imshow(f1.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis")
        labels = [DS_NAMES.get(n, n).replace("NF-", "").replace("-v3", "") for n in f1.index]
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        ax.set_yticks(range(len(labels)), labels if ax is axes[0][0] else [""] * len(labels))
        for i in range(len(labels)):
            for j in range(len(labels)):
                v = f1.iat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v < 0.6 else "black", fontsize=9)
        ax.set_title(MODEL_NAMES.get(m, m)); ax.set_xlabel("tested on")
        if ax is axes[0][0]:
            ax.set_ylabel("trained on")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="F1 (attack class)")
    fig.suptitle("F1, time-ordered split, threshold fixed on the training dataset", fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--readme", type=Path, default=ROOT / "README.md")
    args = ap.parse_args()

    res = pd.read_csv(args.results_dir / "results_all.csv")
    cls = pd.read_csv(args.results_dir / "results_per_class.csv")

    head = headline(res)
    md = ["### Table 1 - Average over dataset pairs (mean ± std over seeds)", "", md_table(head), "",
          "*Oracle F1 uses the best threshold on the test set itself - not achievable in deployment, "
          "shown only to separate threshold miscalibration from loss of signal.*", ""]
    for m in res.model.unique():
        tab, _ = matrix(res, m)
        md += [f"### Table 2{'abc'[list(res.model.unique()).index(m)]} - {MODEL_NAMES.get(m, m)}: F1 (false-positive rate), "
               "time-ordered split", "", md_table(tab.reset_index(drop=True)), ""]
    pc, _ = per_class_table(cls)
    md += ["### Table 3 - Share of test flows flagged as attack, per class (time-ordered split, all models)", "",
           "*Trained same = model trained on the test set's own dataset; trained other = average over models trained "
           "on the other datasets. For Benign this is the false-alarm rate (lower is better).*", "",
           md_table(pc), ""]
    tables = "\n".join(md)
    (args.results_dir / "results_tables.md").write_text(tables)

    kn = key_numbers(res, cls)
    kn["draft_finding"] = draft_finding(kn)
    (args.results_dir / "key_numbers.json").write_text(json.dumps(kn, indent=2))
    heatmaps(res, args.results_dir / "cross_dataset_f1.png")

    if args.readme.exists():
        readme = args.readme.read_text()
        readme = replace_block(readme, "RESULTS", tables + "\n![F1 heatmaps](results/cross_dataset_f1.png)")
        readme = replace_block(readme, "FINDING", "> " + kn["draft_finding"])
        args.readme.write_text(readme)
    print(tables)
    print("\nKEY FINDING (auto-draft - check every number before quoting):\n" + kn["draft_finding"])


if __name__ == "__main__":
    main()
