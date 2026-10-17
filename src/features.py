"""Shared-feature check and feature matrices."""
import numpy as np
import pandas as pd

from .config import excluded_columns


def shared_features(dfs: dict, drop_ttl: bool = False):
    """Numeric columns present in every dataset, minus excluded ones.

    Returns (features, report) where report is a presence/dtype table that is
    written to results/feature_check.csv so the alignment can be inspected.
    """
    excluded = excluded_columns(drop_ttl)
    all_cols = list(dict.fromkeys(c for df in dfs.values() for c in df.columns))
    rows = []
    for c in all_cols:
        row = {"column": c}
        for name, df in dfs.items():
            row[name] = str(df[c].dtype) if c in df.columns else "MISSING"
        present = all(c in df.columns for df in dfs.values())
        numeric = present and all(pd.api.types.is_numeric_dtype(df[c]) for df in dfs.values())
        if c in excluded:
            row["used"], row["reason"] = False, "excluded (identifier / time / label)"
        elif not present:
            row["used"], row["reason"] = False, "not in every dataset"
        elif not numeric:
            row["used"], row["reason"] = False, "non-numeric"
        elif all(df[c].nunique() <= 1 for df in dfs.values()):
            row["used"], row["reason"] = False, "constant everywhere"
        else:
            row["used"], row["reason"] = True, ""
        rows.append(row)
    report = pd.DataFrame(rows)
    feats = report.loc[report["used"], "column"].tolist()
    return feats, report


def to_matrix(df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    X = np.array(df[feats].to_numpy(dtype=np.float64), copy=True)  # writable copy (pandas CoW)
    X[~np.isfinite(X)] = 0.0
    # values above float32 range would overflow; clip before casting
    np.clip(X, -3e38, 3e38, out=X)
    return X.astype(np.float32)


def signed_log1p(X):
    """Compress heavy-tailed byte/packet counts (used before the MLP)."""
    return np.sign(X) * np.log1p(np.abs(X))
