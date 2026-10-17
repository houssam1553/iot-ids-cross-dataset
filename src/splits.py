"""Train/validation/test splits (70 / 15 / 15).

random   : stratified random split - the protocol most papers use. With
           time-ordered traffic it lets near-identical flows from the same attack
           burst land in both train and test, so it is optimistic.
temporal : per class, flows are sorted by FLOW_START_MILLISECONDS; the first 70%
           train, next 15% validate, last 15% test. The model is always tested on
           traffic that happened *after* everything it saw, and every class is
           present in every split (a single global time cut would leave whole
           attack types out of the test set, because attacks ran in time blocks).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ATTACK_COL, TIME_COL

FRACTIONS = (0.70, 0.15, 0.15)


def random_split(df: pd.DataFrame, seed: int):
    counts = df[ATTACK_COL].value_counts()
    strat = df[ATTACK_COL].where(df[ATTACK_COL].map(counts) >= 10, "_rare")
    idx = np.arange(len(df))
    tr, rest = train_test_split(idx, train_size=FRACTIONS[0], stratify=strat, random_state=seed)
    va, te = train_test_split(rest, test_size=0.5, stratify=strat.iloc[rest], random_state=seed)
    return df.iloc[tr], df.iloc[va], df.iloc[te]


def temporal_split(df: pd.DataFrame, seed: int = 0):
    if TIME_COL not in df.columns:
        raise KeyError(f"{TIME_COL} missing - temporal split needs NetFlow v3 data")
    tr, va, te = [], [], []
    for _, g in df.groupby(ATTACK_COL, sort=False):
        order = g.sort_values(TIME_COL, kind="stable").index
        a, b = int(len(order) * FRACTIONS[0]), int(len(order) * (FRACTIONS[0] + FRACTIONS[1]))
        tr.append(order[:a]); va.append(order[a:b]); te.append(order[b:])
    cat = lambda parts: df.loc[np.concatenate(parts)]
    return cat(tr), cat(va), cat(te)


SPLITS = {"random": random_split, "temporal": temporal_split}
