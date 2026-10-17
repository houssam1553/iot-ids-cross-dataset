"""Small helpers for finding, saving and loading data files."""
from pathlib import Path

import pandas as pd

from .config import ATTACK_COL


def find_csv(raw_dir: Path, tokens: list[str]) -> Path | None:
    """Return the first CSV under raw_dir whose lower-case name contains all tokens."""
    for p in sorted(Path(raw_dir).rglob("*.csv")):
        name = p.name.lower()
        if "feature" in name:          # skip NetFlow_v3_Features.csv descriptions
            continue
        if all(t in name for t in tokens):
            return p
    return None


def normalise_attack(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    return s.where(s.str.lower() != "benign", "Benign")


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Markdown table without the optional `tabulate` dependency."""
    fmt = lambda v: floatfmt.format(v) if isinstance(v, float) else str(v)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def save_frame(df: pd.DataFrame, path_no_ext: Path) -> Path:
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    try:
        out = path_no_ext.with_suffix(".parquet")
        df.to_parquet(out, index=False)
    except ImportError:
        out = path_no_ext.with_suffix(".pkl")
        df.to_pickle(out)
    return out


def load_frame(path_no_ext: Path) -> pd.DataFrame:
    for suffix, reader in ((".parquet", pd.read_parquet), (".pkl", pd.read_pickle)):
        p = path_no_ext.with_suffix(suffix)
        if p.exists():
            df = reader(p)
            df[ATTACK_COL] = normalise_attack(df[ATTACK_COL])
            return df
    raise FileNotFoundError(f"No processed file for {path_no_ext} - run `python -m src.prepare` first.")
